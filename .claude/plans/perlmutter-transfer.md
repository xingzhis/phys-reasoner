# Plan: Perlmutter Transfer & Compressed Production Run

**Status:** Active — drafted 2026-04-09
**Deadline:** Workshop submission 2026-04-24 (15 days)
**Owner:** xs272 (test cluster) + collaborator (Perlmutter)
**Related:** `.claude/plans/physcode.md` (parent plan, original 6-week timeline now compressed)

---

## Goal

Transfer the async GRPO training pipeline (currently running on a 3-GPU test HPC: 1 rollout / 1 trainer / 1 verifier) to NERSC Perlmutter (4×A100-80G nodes + 4×A100-40G nodes), and complete one production training + eval run by **2026-04-24** for a workshop submission.

---

## Reality check

15 days minus 3–4 days of cluster bring-up = **~10 effective training-and-eval days**. That means:

- **One** production training config, picked from minimal testing, run long.
- **No** hyperparameter sweep beyond LR (1e-6 vs 3e-6, ~200 steps each).
- **Drop** CoT-GRPO baseline as a parallel run (revisit only if spare nodes appear).
- **Drop** ablations and curriculum stage 2 (mixed types). Numerical-only is the highest-signal subset; train on that, evaluate on everything.
- **Drop** principia, critpt, verifier comparison side-quests entirely.
- **Skip** the full-105k Goldilocks pass@8. Sample 20–30k Dr. SCI rows, run pass@8 on that, take the Goldilocks subset (~4–6k rows), add the 6.8k corpus → ~12k training prompts. Enough for 1000 GRPO steps at batch=64.

If the production run cannot reach ~25% accuracy by step 500, fall back to the **reward-noise framing** (physcode proposal §11) for the workshop submission. Have that draft ready in parallel.

---

## Principles

1. **Don't scale dataset size, scale signal density.** GRPO converges in 500–2000 steps. At `train_batch=32`, `rollout_n=8`, that's ~30k unique-prompt rollouts — ~12k high-quality Goldilocks prompts is enough.
2. **Disaggregated async = step time bound by max(rollout, train).** Long thinking traces make rollout the bottleneck. Optimize rollout first (TP, KV cache, prompt count), trainer second.
3. **Reproduce-then-scale.** Get one identical small run going on both clusters before changing anything. If it differs, you've isolated the cluster, not the config.
4. **Checkpoint frequently, validate rarely.** Small frequent ckpts (every 50 steps) protect against node failures; eval offline only at 250/500/750/1000.
5. **One knob at a time.** Don't co-tune LR + batch + temperature; you can't attribute regressions.

---

## Timeline

| Date | What | Where | Owner |
|---|---|---|---|
| Apr 9–10 | Build Goldilocks-filtered training parquet (~12k); fix xVerify client retry; lock verl SHA | test cluster | xs272 |
| Apr 10 | Real-budget 1-step Qwen3.5-4B run; measure step time, OOM headroom | test cluster | xs272 |
| Apr 11 | 20-step run — get throughput + step-1 reward signal | test cluster | xs272 |
| Apr 11 | Send collaborator questions; start writing `bootstrap_perlmutter.sh` | local | xs272 |
| Apr 12 | `bootstrap_perlmutter.sh` finished; per-job sbatch with verifier baked in; SIF + parquets pushed | local + cluster | xs272 |
| Apr 13 | Collaborator runs bootstrap, runs 0.8B smoke, runs 4B 1-step on Perlmutter | Perlmutter | collaborator (xs272 on call) |
| Apr 14 | Collaborator replicates 20-step run; compare numbers to test cluster | Perlmutter | collaborator |
| Apr 15 | LR sweep: 1e-6 vs 3e-6, 200 steps each | Perlmutter | collaborator |
| Apr 16 | **Launch production run**: numerical-only, 1000 steps, ckpt every 50 | Perlmutter | collaborator (xs272 on call) |
| Apr 16–20 | Production run finishes (~4 days wall-time); checkpoints pulled | Perlmutter | collaborator |
| Apr 20–22 | Eval at ckpts 250/500/750/1000 on dev + MATH-500 hard | Perlmutter or test | xs272 |
| Apr 22–23 | Write paper; freeze numbers | local | xs272 |
| Apr 24 | Submit | — | xs272 |

**Slack budget: zero.** Anything that slips by >1 day → fall back to reward-noise framing.

**xs272 must be on a video call with the collaborator on Apr 13 (bootstrap day) and Apr 16 (production launch).** Block those days now.

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
   - Figure-reference rows (already dropped in `drsci_train.parquet`)
   - Rows whose prompt is >1024 tokens

Curriculum stage 1 = **numerical-only subset** of the above (~7–8k rows). Stage 2 (mixed types) deferred unless time permits.

**Do NOT** add principia-collection or other corpora unless reward signal is sparse (<5% mean reward) at step 200.

---

## Production training config

```bash
MODEL=Qwen/Qwen3.5-4B
TRAIN_BATCH=32           # was 4 in smoke
ROLLOUT_N=8              # was 4
PPO_MICRO_BS_PER_GPU=2   # FSDP across 2 trainer GPUs
LR=1e-6                  # confirm via the 200-step sweep first
THINKING_BUDGET=12288
TOOL_CALL_BUDGET=2048
ANSWER_BUDGET=4096
# → MAX_RESPONSE_LEN ≈ 18900 (existing formula in train_async.sh is correct)
N_GPUS_TRAIN=2
N_GPUS_ROLLOUT=2
ACTOR_PARAM_OFFLOAD=False
ACTOR_OPT_OFFLOAD=False  # 80G is enough for FSDP/2 + AdamW
VLLM_GPU_MEM_UTIL=0.88
TOTAL_STEPS=1000
SAVE_FREQ=50
TEST_FREQ=-1             # eval offline from ckpts, not in-loop
```

**Throughput estimate:**
- Per-step rollout budget: 32 prompts × 8 samples × ~16k tokens = ~4M tokens/step
- vLLM async on 2×A100-80G TP=2: ~50–80k tok/s → ~50–80s/step rollout
- FSDP/2 trainer on 4B bf16 + grad_ckpt: ~20–30s/step
- **Estimated step time: 60–90s. 1000 steps = 17–25 hours.** Fits one 24h job, or two 12h jobs with resume.

If collaborator's Apr 14 test shows step time >120s, fall back: drop `TRAIN_BATCH` to 24, or `ROLLOUT_N` to 6, or shrink `THINKING_BUDGET` to 8192.

---

## Suboptimalities in current `train_async.sh` to fix before production

| # | Current | Issue | Production |
|---|---|---|---|
| 1 | `TRAIN_BATCH=4`, `ROLLOUT_N=4` | Way too small for stable GRPO; gradient noise dominates | `TRAIN_BATCH=32`, `ROLLOUT_N=8` |
| 2 | `MAX_PROMPT_LEN=1024` | Probably fine; verify on full corpus | Keep; drop rows >1024 |
| 3 | `MAX_RESPONSE_LEN=18904` | Correct envelope; smoke shows `clip_ratio=0.94` (thinking hits budget) | Accept clipping (think-interrupt fires) |
| 4 | `LR=1e-6` | On the conservative end | Sweep 1e-6 vs 3e-6 in 200-step study |
| 5 | `temperature=1.0`, `top_p=0.9` | Good for diversity | Keep |
| 6 | `kl_loss_coef=0.001` | Standard | Keep; sweep 0.01 only if reward collapses |
| 7 | `tensor_model_parallel_size=$N_GPUS_ROLLOUT` (=1 in smoke) | Fine on 1 GPU; 2 GPUs need TP=2 for KV headroom | TP=2 on rollout side |
| 8 | `n=4` rollout group | Too small for stable GRPO advantage estimate | n=8 (sweet spot) |
| 9 | `staleness_threshold=0` (on-policy) | Safest, slowest | Start on-policy; flip to staleness=1 + partial_rollout=True only after a stable run |

---

## Resource layout — Perlmutter

**Per-job xVerify** (no persistent service). Two options:

### Option 1 — Heterogeneous SLURM job (preferred if supported)

```
sbatch --het-group=0 --gres=gpu:4 -N1 \  # 4×A100-80G: trainer+rollout
       --het-group=1 --gres=gpu:1 -N1 \  # 1×A100-40G: xVerify-7B
       run_grpo.sbatch
```

- Het-group 1 starts xVerify on `$(hostname)` and writes the URL to a shared file in `$SCRATCH`.
- Het-group 0 reads the URL and exports `XVERIFY_URL=http://<hostname>:8765/judge`.
- Confirm with collaborator (Q2.3 in collaborator-questions doc) whether het jobs work on Perlmutter.

### Option 2 — Single 4×80G node, share one GPU

- GPU 0: trainer (FSDP)
- GPU 1: trainer
- GPU 2: rollout (vLLM, TP=1, gpu_mem_util=0.85)
- GPU 3: rollout (TP=1) + xVerify-7B co-located (vLLM at 0.5, xVerify at 0.4)

Cleaner queue, uglier memory. xVerify-7B fp16 ≈ 14GB. Try Option 1 first; fall back to Option 2 if het jobs aren't supported.

---

## Transfer protocol

### Stage 0 — pre-transfer (xs272, on test cluster)

- [ ] Lock the verl submodule SHA, push to `physcode` branch
- [ ] Push all `scripts/`, `src/`, prompts; tag the commit so collaborator can `git checkout` exactly that
- [ ] Export the Apptainer SIF — SIFs are portable. Overlays may not be → must be rebuildable from script.
- [ ] Write `scripts/bootstrap_perlmutter.sh` that:
  1. Pulls SIF
  2. Builds overlay
  3. Installs phys-extras (transformers 5.3.0, hub 1.8.0, fla)
  4. Installs `.async-extras` (numpy 2.x for cupy)
  5. Runs `e2fsck -fp $OVERLAY`
  6. Runs the test suite (`pytest tests/ -v`)
- [ ] Write `scripts/run_grpo_perlmutter.sbatch` (het job with baked-in xVerify)
- [ ] Pre-stage HF weights (`Qwen/Qwen3.5-4B`, `Qwen/Qwen3.5-0.8B`, `IAAR-Shanghai/xVerify-7B-I`) — instructions for collaborator
- [ ] Push training parquets (or rebuild instructions if too large)

### Stage 1 — bring-up (collaborator on Perlmutter, Apr 13)

- [ ] `bootstrap_perlmutter.sh` → SIF + overlays in place
- [ ] HF cache populated
- [ ] Run `scripts/train_smoke_async.sh` with `MODEL=Qwen/Qwen3.5-0.8B` — confirms environment
- [ ] Run a 4B 1-step job — confirms memory + speed
- [ ] Compare 1-step metrics vs test-cluster numbers; if step time differs by >2×, debug before scaling

### Stage 2 — production (Apr 15–20)

- [ ] LR sweep (200 steps × 2 jobs)
- [ ] Production run: numerical curriculum, 1000 steps, ckpt every 50
- [ ] Optional/parallel (only if budget allows): CoT-GRPO baseline

---

## Failure modes likely mid-training

| Symptom | Likely cause | Fix |
|---|---|---|
| vLLM OOM on long thinking traces | KV cache too big | Lower `gpu_memory_utilization` or `THINKING_BUDGET` |
| xVerify server crashes mid-run | No retry | Implement xverify_client retry (TODO #1 in session-notes verifier section) — **must do before transfer** |
| Ray actor death from preempted node | Node preemption | Checkpoint every 50 steps; document resume procedure |
| WandB auth failure on compute node | Outbound HTTPS blocked | Fall back to console logging, scrape logs after |
| Driver / CUDA mismatch with SIF | Different CUDA on Perlmutter | Test on Day 1 (Apr 13), not Day 5 |
| HF API rate-limit | Concurrent downloads | Always `local_files_only=True` after initial download |
| numpy/cupy ABI fail in async path | `.async-extras` paths differ | Rebuild `.async-extras` on Perlmutter from same SIF |
| Step time >2× test-cluster | Network, NUMA, BIOS, ECC scrub | Profile first; do not scale |

---

## Concrete TODO list

### On test cluster (Apr 9–12)

1. [ ] Run pass@8 over 25k Dr. SCI sample → produce Goldilocks-filtered ~5k parquet
2. [ ] Build production training parquet (Goldilocks + corpus, 1k dev holdout)
3. [ ] Real-budget smoke test (Qwen3.5-4B, 1 step) — get step time number
4. [ ] 20-step run — get throughput number, confirm reward signal ≥0.05
5. [ ] Implement xVerify client retry (session-notes verifier TODO #1)
6. [ ] Write `bootstrap_perlmutter.sh`
7. [ ] Write `run_grpo_perlmutter.sbatch` (het job)
8. [ ] Write `serve_xverify_perlmutter.sbatch` (or bake into the het job's group 1)

### On Perlmutter (collaborator, after #1–8 done)

9. [ ] Bootstrap, run test suite
10. [ ] Replicate 20-step run; compare numbers
11. [ ] LR sweep (200 steps × 2)
12. [ ] Numerical-only 1000-step production run
13. [ ] (Optional) CoT-GRPO baseline run
14. [ ] MATH-500 hard eval after each ckpt

xs272 should be paired with collaborator for #9 (bootstrap) and #12 (production launch).

---

## Open decisions (need answers before Apr 12)

1. CoT-GRPO baseline: include or drop? (Doubles node-hours)
2. Whether collaborator can support a het SLURM job (Option 1) or must use single-node (Option 2)
3. Actual wall-time limit (drives ckpt-and-resume design)
4. WandB or console-only logging

See `docs/perlmutter-collaborator-questions.md` for the full question list to send.
