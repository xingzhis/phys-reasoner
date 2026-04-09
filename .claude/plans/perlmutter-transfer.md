# Plan: Perlmutter Transfer & Compressed Production Run

**Status:** Active — drafted 2026-04-09, revised 2026-04-09 (multi-node rollout)
**Deadline:** Workshop submission 2026-04-24 (15 days)
**Owner:** xs272 (test cluster) + collaborator (Perlmutter)
**Related:**
- `.claude/plans/physcode.md` (parent plan; original 6-week timeline now compressed)
- `docs/perlmutter-profiling.md` (profiling methodology + bottleneck diagnosis)
- `docs/perlmutter-collaborator-questions.md` (open questions for collaborator)

---

## Goal

Transfer the GRPO TIR training pipeline (currently smoke-tested on a 3-GPU test HPC: 1 rollout / 1 trainer / 1 verifier, async path) to NERSC Perlmutter (4×A100-80G + 4×A100-40G nodes), and complete one production training + eval run by **2026-04-24** for a workshop submission.

---

## Reality check

15 days minus 3–4 days of cluster bring-up = **~10 effective training-and-eval days**. Calendar is the binding constraint; compute is not. Collaborator has confirmed 3000 node-hours of budget — production runs are on the order of **30–50 node-hours each**, so the budget supports many runs in parallel. The strategy is therefore:

- **Use compute to buy resilience and statistical power**, not to scale model size or dataset size.
- Run things in parallel; treat queue contention as the bottleneck, not GPU-hours.
- Reserve 50% of the budget as failure buffer — first runs on a new cluster always need retries.
- Pick configurations that minimize **wall-clock time per run** even if they cost more node-hours.

Calendar-bounded scope decisions:
- ❌ Multi-seed promoted to base — too much eval load. Seed 2 is **conditional** on seed 1 telemetry at step 100.
- ❌ 9B model — 2.5× the compute, new bug surface, doesn't fit one 48h job, marginal scientific gain. Save for journal version.
- ❌ Hyperparameter sweep beyond a 2-point LR check (1e-6 vs 3e-6).
- ❌ Curriculum stage 2 (mixed types). Numerical-only is the highest-signal subset.
- ❌ Verifier comparison, principia, critpt — all post-workshop.
- ✅ CoT-GRPO baseline as a parallel run (1 seed). Compute is free; this is the primary comparison.
- ✅ 0.8B TIR-GRPO scaling run (cheap, ~12h wall-time, gives a 2-point scaling story).

If the production run cannot reach ~25% accuracy by step 500, fall back to the **reward-noise framing** (physcode proposal §11). Have that draft ready in parallel.

---

## Principles

1. **Compute is free; calendar is the constraint.** Optimize wall-clock per run, not node-hours per run.
2. **Don't scale dataset size, scale signal density.** GRPO converges in 500–2000 steps. ~12k Goldilocks-filtered prompts is enough.
3. **For long-context rollouts, KV cache is the bottleneck.** More rollout GPUs (TP within node, DP across nodes) buys both throughput and concurrent-sequence capacity. See "Resource layouts" below.
4. **Cross-node TP and cross-node FSDP are killers; cross-node DP rollout is fine.** Slingshot 11 bandwidth is enough for parameter sync but not for per-layer all-reduce. See "Network rationale" below.
5. **Reproduce-then-scale.** Get one identical small run going on both clusters before changing anything. If results differ, you've isolated the cluster, not the config.
6. **Checkpoint frequently, validate rarely.** `SAVE_FREQ=50` (~75 min of wall-time per ckpt); eval offline at 250/500/750/1000 only.
7. **One knob at a time.** Don't co-tune LR + batch + temperature; you can't attribute regressions.
8. **Bracket bets before committing.** When two configs are plausible and you can't predict which wins, run both small and pick the winner.

---

## Resource layouts under consideration

For Qwen3.5-4B with 16k-token thinking traces, batch=32, n=8 (≈4M tokens/step):

| Config | Nodes | Trainer | Rollout | Verifier | Est. step time | 1000-step wall-h | Node-h/run |
|---|---|---|---|---|---|---|---|
| **A. Single-node colocate** | 1×80G + 0.25 (shared) | 4 GPU FSDP/4 (serial w/ rollout) | 4 GPU TP=4 (serial w/ trainer) | 1 GPU shared QoS | ~85s | ~24h | ~30 |
| **B. 1+1 async** (dominated by A) | 2×80G + 0.25 | 4 GPU FSDP/4 | 4 GPU TP=4 | 1 GPU shared QoS | ~75s | ~21h | ~47 |
| **C. 1+2 async, DP=2 rollout** | 3×80G + 0.25 | 4 GPU FSDP/4 | 2× (4 GPU TP=4) DP engines | 1 GPU shared QoS | ~45s | ~13h | ~42 |
| D. 1+4 async, DP=4 rollout | 5×80G + 0.25 | 4 GPU FSDP/4 | 4× (4 GPU TP=4) DP engines | 1 GPU shared QoS | ~25s (trainer-bound) | ~7h | ~37 |

**Selection logic:**
- Config B is dominated by A (same TP, more nodes, no real overlap when on-policy). Drop.
- Config A is the conservative single-node default (least failure modes).
- **Config C is the recommended production layout**: ~47% faster wall-clock than A, only ~12 more node-hours per run, sub-7% of total budget per production run. Trainer is no longer dominant but not yet bottleneck.
- Config D adds nodes for nothing — the trainer becomes the bottleneck and rollout speedup is wasted (we explicitly do not FSDP across nodes for a 4B model).

**Decision: Config C is the production target. Config A is the validated fallback.** Both are tested side-by-side on Apr 13 (the Apr 13 bracket test) before committing for production.

### Network rationale (why these layouts and not others)

Perlmutter Slingshot 11 = 4×200 Gbps NICs per node = ~100 GB/s per node off-node bandwidth.
Within a node: NVLink at ~600 GB/s between A100s.

| Communication pattern | Volume per step | Where it lives | Verdict |
|---|---|---|---|
| TP all-reduce per attention layer | ~60 MB × thousands of layer-calls per sequence × 256 sequences | NVLink only (within node) | Fine within node; **catastrophic across nodes** |
| FSDP all-gather/reduce-scatter, 4B model | ~8 GB total per step | NVLink (single-node FSDP/4) | Fine. Cross-node FSDP would add ~10s/step — not worth it for 4B |
| DP rollout: param sync from trainer | ~8 GB per sync interval | Slingshot, amortized over many steps | Negligible |
| DP rollout: prompts in / tokens out | A few MB | Slingshot | Negligible |

**Rules for this model size on this cluster:**
- Trainer: stay single-node, FSDP/4. Do **not** FSDP across nodes for a 4B model.
- Rollout: scale across nodes via DP (multiple vLLM engines), each with TP=4 within its node. Do **not** TP across nodes — Slingshot bandwidth cannot keep up with per-attention-layer all-reduces.
- Verifier: 1 GPU on `shared` QoS (separate small allocation). xVerify-7B fp16 ≈ 14 GB, fits one A100.

---

## Production training config (Config C)

```bash
MODEL=Qwen/Qwen3.5-4B
TRAIN_BATCH=32           # ppo_mini_batch_size
ROLLOUT_N=8              # GRPO group size
PPO_MICRO_BS_PER_GPU=2   # FSDP/4 trainer
LR=1e-6                  # confirm via Apr 15 LR sweep
THINKING_BUDGET=12288
TOOL_CALL_BUDGET=2048
ANSWER_BUDGET=4096
# → MAX_RESPONSE_LEN ≈ 18900
N_GPUS_TRAIN=4           # 4 GPUs on the trainer node, FSDP/4
N_GPUS_ROLLOUT=4         # 4 GPUs per rollout node, TP=4
NNODES_TRAIN=1
NNODES_ROLLOUT=2         # ← Config C: 2 rollout nodes with DP=2
ACTOR_PARAM_OFFLOAD=False
ACTOR_OPT_OFFLOAD=False  # Sharded AdamW across 4 GPUs fits on 80G
VLLM_GPU_MEM_UTIL=0.88
TOTAL_STEPS=1000
SAVE_FREQ=50
TEST_FREQ=-1             # eval offline from ckpts
```

Memory check (per GPU, FSDP/4 trainer node):

| Component | Per GPU |
|---|---|
| Model weights bf16 / 4 | 2 GB |
| Gradients bf16 / 4 | 2 GB |
| AdamW state fp32 × 2 / 4 | 4 GB |
| Activations + framework | ~10 GB |
| **Total trainer-side** | **~18 GB** |

Rollout-side per GPU (TP=4): vLLM weights ~2 GB shard + KV cache ~50 GB at `gpu_mem_util=0.88` + framework ~10 GB ≈ 62 GB. Comfortable.

**Throughput estimate (Config C):**
- Per-step rollout: 4M tokens, served by 2 vLLM engines DP=2 × TP=4. Each engine: ~50k tok/s on Qwen3.5-4B at long context → ~40s rollout per step.
- Trainer: 4B FSDP/4, batch 32, ppo: ~15s per step.
- Async on-policy = serial: rollout (40) + train (15) = ~45–55s per step.
- 1000 steps ≈ 13h wall-time, fits one 24h job comfortably.

**Estimates are guesses until Apr 14 measurement.** If Config C step time >70s, the bottleneck profiling protocol (`docs/perlmutter-profiling.md`) drives the next change.

---

## Configuration changes from current `train_async.sh`

| # | Current | Production (Config C) |
|---|---|---|
| 1 | `TRAIN_BATCH=4`, `ROLLOUT_N=4` | `TRAIN_BATCH=32`, `ROLLOUT_N=8` |
| 2 | `N_GPUS_TRAIN=1`, `N_GPUS_ROLLOUT=1` | `N_GPUS_TRAIN=4`, `N_GPUS_ROLLOUT=4` |
| 3 | `NNODES_TRAIN=1`, `NNODES_ROLLOUT=1` | `NNODES_TRAIN=1`, `NNODES_ROLLOUT=2` (DP=2) |
| 4 | `tensor_model_parallel_size=1` (single GPU rollout) | `tensor_model_parallel_size=4` |
| 5 | `MAX_PROMPT_LEN=1024`, `MAX_RESPONSE_LEN≈18900` | Same (already correct) |
| 6 | `temperature=1.0`, `top_p=0.9`, `kl_loss_coef=0.001` | Same |
| 7 | `LR=1e-6` | Confirm via Apr 15 LR sweep (1e-6 vs 3e-6) |
| 8 | `staleness_threshold=0` (on-policy) | Start same; flip to staleness=1 only after a stable run |
| 9 | `SAVE_FREQ=-1` | `SAVE_FREQ=50` |
| 10 | `verifier on dedicated 40G node` (Option 1) | `verifier on shared QoS, 1 GPU` |

---

## Per-job xVerify layout

Het SLURM job, 3 het-groups for Config C:

```bash
sbatch \
  --het-group=0 -N1 --gres=gpu:4 --qos=regular_g  \  # trainer
  : \
  --het-group=1 -N2 --gres=gpu:4 --qos=regular_g  \  # rollout (2 nodes, DP=2 × TP=4)
  : \
  --het-group=2 -N1 --gres=gpu:1 --qos=shared_g   \  # xVerify-7B, 1 GPU shared
  run_grpo_perlmutter.sbatch
```

For Config A (fallback): drop het-group 1, set `NNODES_ROLLOUT=0`, colocate trainer + rollout on the het-group 0 node via the colocate path (`scripts/train.sh`).

Verifier startup:
- Het-group 2 starts xVerify on `$(hostname)` and writes `XVERIFY_URL` to a shared file in `$PSCRATCH`.
- Het-groups 0 and 1 wait on the URL file (max 60s) before launching training.

**Confirm with collaborator** (already in questions doc):
- `shared_g` QoS exists for GPU nodes
- Het jobs can mix `regular_g` and `shared_g` in the same allocation
- 3-group het jobs work (some clusters cap at 2 groups)

---

## Profiling and bottleneck identification

**Goal:** after every job (especially on Apr 13–14), produce a one-table per-phase decomposition of step time and identify the dominant phase. Then compare against the matrix in `docs/perlmutter-profiling.md` to decide what to change next.

**Profiling sources** (all already exist; we just need to read them):
1. **VeRL `timing_s/*` metrics** in train.log per step — gen, ref, old_log_prob, update_actor, reward, total
2. **vLLM stats** in train.log — tokens/s, KV cache util %, preempted requests count, queue depths
3. **`nvidia-smi dmon`** snapshot every 2s on a side terminal during the run — GPU util %, memory used, NVLink throughput
4. **`ray status`** for placement group health
5. **Slingshot counters** (need collaborator help — added to questions doc)

**Profiling helper script** (TODO #9 below): `scripts/parse_train_log.py` reads a train.log, extracts `timing_s/*` per step (skipping warmup steps 0–4), and prints a Markdown table:

```
Phase            mean    p50     p95     % of total
gen              42.1s   41.8s   45.3s   78%
ref              0.5s    0.5s    0.6s    1%
old_log_prob     2.1s    2.0s    2.3s    4%
update_actor     8.2s    8.0s    9.1s    15%
reward           0.4s    0.4s    0.5s    1%
total            53.8s   53.1s   57.8s   100%
```

The dominant phase is then matched against the diagnosis matrix in `docs/perlmutter-profiling.md`.

---

## Timeline

| Date | What | Where | Owner |
|---|---|---|---|
| **Apr 9–10** | Build Goldilocks-filtered training parquet (~12k); fix xVerify client retry; lock verl SHA; write profiling helper | test cluster | xs272 |
| **Apr 10** | Real-budget 1-step Qwen3.5-4B on test cluster (single-GPU async, smoke); profile | test cluster | xs272 |
| **Apr 11** | 20-step async smoke run on test cluster; profile, get reward signal | test cluster | xs272 |
| **Apr 11** | Send collaborator questions; start writing `bootstrap_perlmutter.sh` | local | xs272 |
| **Apr 12** | `bootstrap_perlmutter.sh`, `run_grpo_A.sbatch`, `run_grpo_C.sbatch`, `serve_xverify.sbatch` written; SIF + parquets pushed | local + cluster | xs272 |
| **Apr 13 morning** | Collaborator runs bootstrap, 0.8B smoke, 4B 1-step | Perlmutter | collaborator (xs272 on call) |
| **Apr 13 afternoon** | **Apr 13 bracket test**: Config A and Config C side-by-side, 20 steps each, parse `timing_s/*` decomposition, pick winner | Perlmutter | collaborator (xs272 on call) |
| **Apr 14** | Replicate winner with full 50-step run; confirm step time and reward signal stable | Perlmutter | collaborator |
| **Apr 15** | LR sweep: 1e-6 vs 3e-6, 200 steps each, two parallel jobs | Perlmutter | collaborator |
| **Apr 16 morning** | Pick LR; **launch production runs in parallel**: TIR-GRPO seed 1 + CoT-GRPO seed 1 + 0.8B TIR scaling | Perlmutter | collaborator (xs272 on call) |
| **Apr 16 evening** | Inspect ckpt 100 of TIR seed 1 → conditional decision on seed 2 | xs272 | xs272 |
| **Apr 16–19** | Production runs finish (~13–18h wall-time per run); checkpoints pulled | Perlmutter | collaborator |
| **Apr 19–21** | Eval at ckpts 250/500/750/1000 on dev + MATH-500 hard | Perlmutter or test | xs272 |
| **Apr 21–23** | Write paper; freeze numbers | local | xs272 |
| **Apr 24** | Submit | — | xs272 |

**Slack budget: 1–2 days** (better than the previous "zero" thanks to compute headroom). Anything that slips by >2 days → reward-noise fallback.

**xs272 must be paired with collaborator on Apr 13 (bracket test) and Apr 16 (production launch).** Block those days.

---

## Dataset plan

Build a single ~12k training parquet, fixed before training starts.

1. **Sample 25k Dr. SCI rows** (stratified by `inferred_answer_type`).
2. **Pass@8 with Qwen3.5-4B (thinking on)** on that sample → ~6h on H200, run overnight.
3. **Goldilocks filter**: keep rows in [0.15, 0.85] pass@8 band. Expected yield: ~4–6k rows.
4. **Add the 6.8k curated corpus** without filtering — small and high-quality.
5. **Hold out 1k stratified dev set** before training. Never train on it.
6. **Drop entirely**:
   - Rows where rule-verifier round-trip FNR > 50% for that answer type
   - Figure-reference rows (already dropped)
   - Rows whose prompt is >1024 tokens

Curriculum stage 1 = **numerical-only subset** (~7–8k rows). Stage 2 deferred unless time permits.

**Do NOT** add principia-collection or other corpora unless reward signal is sparse (<5% mean reward) at step 200.

---

## Transfer protocol

### Stage 0 — pre-transfer (xs272, on test cluster, Apr 9–12)

- [ ] Lock the verl submodule SHA, push to `physcode` branch
- [ ] Push all `scripts/`, `src/`, prompts; tag the commit so collaborator can `git checkout` exactly that
- [ ] Export the Apptainer SIF — SIFs are portable; overlays are not
- [ ] Write `scripts/bootstrap_perlmutter.sh` that:
  1. Pulls SIF
  2. Builds overlay
  3. Installs phys-extras (transformers 5.3.0, hub 1.8.0, fla)
  4. Installs `.async-extras` (numpy 2.x for cupy)
  5. Runs `e2fsck -fp $OVERLAY`
  6. Runs the test suite (`pytest tests/ -v`)
- [ ] Write `scripts/run_grpo_A_perlmutter.sbatch` (Config A: single-node colocate, 2-group het)
- [ ] Write `scripts/run_grpo_C_perlmutter.sbatch` (Config C: 1+2 multi-node async, 3-group het)
- [ ] Pre-stage HF weights instructions (`Qwen/Qwen3.5-4B`, `Qwen/Qwen3.5-0.8B`, `IAAR-Shanghai/xVerify-7B-I`)
- [ ] Push training parquets

### Stage 1 — bring-up (collaborator on Perlmutter, Apr 13)

- [ ] `bootstrap_perlmutter.sh` → SIF + overlays in place
- [ ] HF cache populated
- [ ] Run `scripts/train_smoke_async.sh` with `MODEL=Qwen/Qwen3.5-0.8B` — confirms environment
- [ ] Run a 4B 1-step job — confirms memory + speed
- [ ] **Apr 13 bracket test**: submit Config A (20 steps) and Config C (20 steps) as parallel jobs; parse `timing_s/*` decomposition for each; commit to one for production

### Stage 2 — production (Apr 14–20)

- [ ] Apr 14: 50-step replication of the winning config; confirm step time + reward stable
- [ ] Apr 15: LR sweep (200 steps × 2 jobs)
- [ ] Apr 16 morning: launch TIR-GRPO seed 1 + CoT-GRPO seed 1 + 0.8B TIR scaling (3 parallel jobs)
- [ ] Apr 16 evening: conditional TIR seed 2 launch based on seed 1 ckpt 100
- [ ] Apr 19–21: eval

---

## Failure modes likely mid-training

| Symptom | Likely cause | Fix |
|---|---|---|
| vLLM OOM on long thinking traces | KV cache too big | Lower `gpu_memory_utilization` or `THINKING_BUDGET` |
| xVerify server crashes mid-run | No retry | Implement xverify_client retry — **must do before transfer** |
| Ray cross-node placement group fails | Ray multi-node misconfig in VeRL fully_async | Fall back to Config A (single node) — pre-tested |
| Trainer waits forever on rollout (Config C) | DP rollout engine hung; one rollout node OOM'd | Inspect logs for the silent engine; restart from ckpt |
| Param sync slow across nodes (Config C) | Slingshot routing problem | Profile with `nvidia-smi dmon`; if persistent, fall back to Config A |
| Step time on Perlmutter >2× the test-cluster baseline | NUMA pinning, ECC scrub, missed env var | Profile FIRST; do not scale prematurely |
| Ray actor death from preempted node | Node preemption | Checkpoint every 50 steps; resume |
| WandB auth failure on compute node | Outbound HTTPS blocked | Fall back to console logging, scrape logs after |
| HF API rate-limit | Concurrent downloads | Always `local_files_only=True` after initial download |
| numpy/cupy ABI fail in async path | `.async-extras` paths differ | Rebuild `.async-extras` on Perlmutter from same SIF |

---

## TODO list

### On test cluster / local (Apr 9–12)

1. [ ] Run pass@8 over 25k Dr. SCI sample → produce Goldilocks-filtered ~5k parquet
2. [ ] Build production training parquet (Goldilocks + corpus, 1k dev holdout)
3. [ ] Real-budget 1-step Qwen3.5-4B test on test cluster (smoke); record step time
4. [ ] 20-step async smoke run on test cluster — get reward signal ≥0.05
5. [ ] Implement xVerify client retry (session-notes verifier TODO #1)
6. [ ] Test checkpoint-and-resume on test cluster (20 → kill → resume → 20 more; metrics continuous?)
7. [ ] Write `scripts/bootstrap_perlmutter.sh`
8. [ ] Write `scripts/run_grpo_A_perlmutter.sbatch` (Config A: single-node colocate, 2-group het)
9. [ ] Write `scripts/run_grpo_C_perlmutter.sbatch` (Config C: 1 trainer + 2 rollout, 3-group het)
10. [ ] Write `scripts/serve_xverify_perlmutter.sbatch` OR bake into the het-group 2 setup
11. [ ] Write `scripts/parse_train_log.py` profiling helper (per-phase timing decomposition)
12. [ ] Write `scripts/eval_ckpts.py` offline eval harness (loads ckpt, runs dev + MATH-500 hard)
13. [ ] Verify `scripts/train.sh` (colocate path) has Qwen3.5 overrides: `attn_implementation:sdpa`, `wrap_policy.transformer_layer_cls_to_wrap=[Qwen3_5DecoderLayer]`, ref offload
14. [ ] Add per-type reward logging to reward function (so train.log shows `reward/numerical/mean`, `reward/expression/mean`, etc.)

### Probes (cheap, run before Perlmutter access if time permits)

15. [ ] Tool-call rate probe on production prompt (50 rollouts, dump_rollouts.py): target >70% emit `<tool_call>`
16. [ ] Thinking-length distribution on same dump: % rollouts hitting `THINKING_BUDGET`
17. [ ] Reward-density probe on Goldilocks subset: % prompts with ≥1/8 correct samples (target >50%)
18. [ ] Verifier FNR on dev set (gold-vs-gold round-trip per type)
19. [ ] Prompt-length histogram (confirm <1% rows >1024 tokens)
20. [ ] Verifier latency under load: simulate 256 reqs / 80s vs xVerify

### On Perlmutter (collaborator, Apr 13+)

21. [ ] Bootstrap, run test suite
22. [ ] 0.8B smoke test
23. [ ] 4B 1-step test
24. [ ] **Apr 13 bracket test**: Config A and Config C, 20 steps each, parse profiling
25. [ ] Apr 14: 50-step replication of winning config
26. [ ] Apr 15: LR sweep
27. [ ] Apr 16: TIR seed 1 + CoT seed 1 + 0.8B scaling, parallel
28. [ ] Apr 16 evening: conditional TIR seed 2 launch
29. [ ] Apr 19–21: offline eval at ckpts

xs272 should be paired with collaborator for Apr 13 (bracket) and Apr 16 (production launch).

---

## Resolved (from NERSC docs + collaborator confirmations 2026-04-09)

- **Wall-time cap: 48h**. 1000 prod steps at 50–90s/step ≈ 14–25h fits one job. Still implement checkpoint-and-resume for preemption safety.
- **Heterogeneous jobs supported.** Going with het-job layout (3 groups for Config C).
- **Compute node internet access**: confirmed.
- **Multiple parallel jobs allowed**: confirmed.
- **Budget**: 3000 node-hours.

## Still open (waiting on collaborator)

1. `shared_g` QoS available for GPU nodes? Mixable with `regular_g` in het jobs?
2. Backfill sweet-spot wall-time (12h vs 24h vs 48h request)
3. Apptainer version + overlay support quirks on Perlmutter
4. Account / partition / qos string for sbatch
5. WandB outbound HTTPS reachability (we have internet, but specific endpoint may be blocked)
6. Slingshot counters / `nvidia-smi nvlink` access on compute nodes (for profiling Config C cross-node)
7. Pair-availability on Apr 13 and Apr 16
8. `$PSCRATCH` size and purge policy

See `docs/perlmutter-collaborator-questions.md`.

## Budget commitment

Assuming Config C wins the bracket test:

| Item | Wall-h | Nodes | Node-h |
|---|---|---|---|
| Apr 13 bracket: A + C × 20 steps | ~6 (parallel) | 1.25 + 3.25 | ~10 |
| Apr 14 replication: 50 steps | ~3 | 3.25 | ~10 |
| Apr 15 LR sweep: 2 × 200 steps | ~5 (parallel) | 6.5 | ~32 |
| Apr 16: TIR seed 1 production | ~13 | 3.25 | ~42 |
| Apr 16: CoT seed 1 production | ~13 | 3.25 | ~42 |
| Apr 16: 0.8B scaling | ~6 | 3.25 | ~20 |
| Conditional: TIR seed 2 | ~13 | 3.25 | ~42 |
| Eval (offline) | ~12 | 1 | ~12 |
| Failure buffer (50%) | — | — | ~110 |
| **Total committed** | | | **~320** |

~10% of 3000. Massive headroom remains for retries and any stretch ablations Apr 19+ if calendar permits.
