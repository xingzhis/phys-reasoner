# Perlmutter Smoke Handoff — Qwen3 Switch

**Target session:** executing the Qwen3-switch smoke on Perlmutter.
**Status:** branches ready locally; untested on Perlmutter.
**Core goal:** validate ≤5 min/step (vs prior 27 min/step OOM baseline).

Read this top-to-bottom once. References at the bottom.

---

## 0. TL;DR

```bash
# On Perlmutter, repo root:
git fetch origin
git -C verl fetch origin

# Option A — full stack (Qwen3 + wandb pin):
git checkout switch/qwen3-thinking
git -C verl checkout switch/qwen3-thinking

# Option B — only wandb pin (if Qwen3 causes issues):
git checkout feat/trainer-node-pin
git -C verl checkout feat/trainer-node-pin

# Sanity check the branch is what you expect:
git log --oneline -5
git -C verl log --oneline -3

# Pre-flight: download model if not cached (see §2):
bash scripts/perlmutter/bootstrap.sh

# Submit smoke:
sbatch scripts/perlmutter/smoke_tir_het.sbatch
```

---

## 1. What changed vs main (summary)

Three sets of changes, stacked on two branches.

### 1a. `feat/trainer-node-pin` (both repos)

Fixes: wandb GPU-metric telemetry reported wrong node under `fully_async_policy`
because the trainer coordinator Ray actor was scheduled on a rollout worker.

- Parent: `scripts/perlmutter/_ray_bringup.sh`, `scripts/train_async_2node.sbatch`
  advertise a `trainer_node: 1` Ray custom resource on the trainer host.
- Verl: `fully_async_main.py` requests `resources={"trainer_node": 0.001}` when
  starting the `FullyAsyncTrainer` actor, with fallback if the resource isn't
  advertised. `fully_async_trainer.py` has a debug `print` of `socket.gethostname()`
  for placement verification (remove after validation).
- Also bundles `perlmutter_debug/FINDINGS.md` (§6, investigation report).

**Known state:** tested on Roberts (wandb now shows trainer GPUs correctly).
Not yet verified on Perlmutter's Slingshot + SLURM resource tagging.

### 1b. `switch/qwen3-thinking` (parent repo; verl branch is empty ref to 1a)

Migrates from Qwen3.5-4B to Qwen3-4B-Thinking-2507 to unlock flash-attn +
Ulysses SP. Unblocks the 2-3× structural slowdown documented in FINDINGS §7.

**`scripts/train_async.sh` edits:**
- Default `MODEL` → `Qwen/Qwen3-4B-Thinking-2507`.
- Default `PPO_MAX_TOKEN_LEN_PER_GPU` 24576 → **20480** (matches
  `max_prompt + max_response`; retool/dapo heuristic).
- **Removed** `+model.override_config.attn_implementation=sdpa` (Qwen3 supports
  flash-attn natively; SDPA was a Qwen3.5-GDN-specific workaround).
- **Removed** both `+*.fsdp_config.wrap_policy.transformer_layer_cls_to_wrap=
  [Qwen3_5DecoderLayer]` overrides.
- **Added** `actor.ulysses_sequence_parallel_size=${TRAIN_SP:-2}` and the
  matching `ref.` line. SP=2 (not 4) on 4 trainer GPUs so DP=2 stays for
  FSDP param sharding.
- **Added** `ref.log_prob_use_dynamic_bsz=True`, `ref.log_prob_max_token_len_per_gpu=
  ${REF_LOG_PROB_MAX_TOKEN_LEN:-81920}` (4× actor budget; retool heuristic).
- **Added** `rollout.log_prob_use_dynamic_bsz=True`.
- **Added** `rollout.max_num_batched_tokens=$((MAX_PROMPT_LEN + MAX_RESPONSE_LEN))`.
- `multi_turn.format=qwen3_coder` → `hermes` (verified: Qwen3-4B-Thinking-2507's
  chat template uses Hermes-style JSON tool calls, not the Qwen3-Coder XML format).
- `async_training.staleness_threshold=0` → `=${STALENESS:-0}` (env-var exposed).

**`scripts/perlmutter/smoke_tir_het.sbatch` edits:**
- `PYTORCH_ALLOC_CONF=expandable_segments:True` in the apt `--env` list.
- Default `MODEL` → `Qwen3-4B-Thinking-2507`.
- Default `PPO_MAX_TOKEN_LEN_PER_GPU` → 20480.
- Defaults for `THINKING_BUDGET`/`TOOL_CALL_BUDGET`/`ANSWER_BUDGET` blanked
  (think-interrupt disabled; model respects `max_response_length` as the hard
  cap). Verified: empty env-vars → Hydra key omitted (`${VAR:+...}`) →
  `thinking_budget=None` inside `tool_agent_loop.py` → interrupt code path
  entirely bypassed, zero overhead.
- Default `TRAIN_SP=2`, `STALENESS=1` (overlap gen/train for ~12% wall savings).

**Not touched:** reward function, TIR prompts, dataset, rollout topology
(still 5 rollout nodes × 4 GPU + 1 trainer node × 4 GPU; shrink rollout
post-smoke if time permits per FINDINGS §12 #4).

---

## 2. Pre-flight on Perlmutter

### 2.1 Branch checkout

```bash
cd $ROOT  # Perlmutter repo root (collaborator's path, likely /pscratch/sd/b/bwhou/17-phy-reasoner/phys-reasoner)
git fetch origin
git -C verl fetch origin
git checkout switch/qwen3-thinking
git -C verl checkout switch/qwen3-thinking

# Confirm:
git log --oneline --decorate -5
# Expect HEAD → switch/qwen3-thinking → 2 commits → feat/trainer-node-pin → main
git -C verl log --oneline --decorate -3
# Expect HEAD → switch/qwen3-thinking == feat/trainer-node-pin (1 commit above physcode)
```

### 2.2 Sanity checks before `sbatch`

```bash
# Apptainer images exist and match env.sh (env.sh is source of truth — defaults
# to -017.img on Roberts; if Perlmutter ships -017b.img, override OVERLAY in .env):
ls $ROOT/verl_vllm017.latest.sif "$OVERLAY"

# Data files in place:
ls $ROOT/data/processed_tir/data/{train,validation}.parquet

# PYTHONPATH leak from collaborator's shell (FINDINGS §2, non-fatal warning):
echo $PYTHONPATH
# If it contains DeepGWBSE/EXPH/BGWAgent paths, `unset PYTHONPATH` before sbatch
# OR add `unset PYTHONPATH` to env.sh. Apptainer refuses to forward these (good),
# but inter-srun steps can still inherit them.

# WANDB and HF tokens:
test -n "$WANDB_API_KEY" && echo "wandb ok" || echo "set WANDB_API_KEY"
test -f ~/.cache/huggingface/token && echo "hf ok"

# TODO(collab) markers in the sbatch resolved?
grep -n 'TODO(collab)' scripts/perlmutter/smoke_tir_het.sbatch
# Expect: 0 results, or resolved with SBATCH_ACCOUNT and qos values.
```

### 2.3 Model download (one-time)

The model cache is under `$HF_HOME` on Perlmutter (typically `$ROOT/hf_cache`).
Qwen3-4B-Thinking-2507 is not pre-cached; download before submitting:

```bash
ROOT=<perlmutter repo root>
source "$ROOT/env.sh"  # provides $SIF, $OVERLAY, $HF_HOME
PYTHONNOUSERSITE=1 apptainer exec \
  --overlay "$OVERLAY:ro" --no-home \
  --bind /etc/pki:/etc/pki \
  --env "PYTHONNOUSERSITE=1" --env "HF_HOME=$HF_HOME" \
  "$SIF" \
  huggingface-cli download Qwen/Qwen3-4B-Thinking-2507

# Verify:
ls $HF_HOME/models--Qwen--Qwen3-4B-Thinking-2507/snapshots/*/
# Expect config.json, tokenizer_config.json, model*.safetensors
```

Do the same for `Qwen/Qwen3-4B-Instruct-2507` if you want the option to swap
(one-line env override) without resubmitting.

### 2.4 Chat template quick-check (verifies hermes format assumption)

```bash
source "$ROOT/env.sh"  # provides $SIF, $OVERLAY, $HF_HOME
PYTHONNOUSERSITE=1 apptainer exec \
  --overlay "$OVERLAY:ro" \
  --env "PYTHONPATH=/opt/phys-extras/" --env "HF_HOME=$HF_HOME" \
  "$SIF" \
  python3 -c "
from transformers import AutoTokenizer
t = AutoTokenizer.from_pretrained('Qwen/Qwen3-4B-Thinking-2507', local_files_only=True)
msgs = [{'role':'user','content':'What is 2+2?'}]
out = t.apply_chat_template(msgs, tools=[{'type':'function','function':{'name':'python','description':'Run python','parameters':{'type':'object','properties':{}}}}], tokenize=False, add_generation_prompt=True)
print(repr(out[-800:]))"
# Expect output to contain:
#   <think> ... </think> (thinking wrapper is hardcoded for Thinking-2507)
#   <tool_call> {...json...} </tool_call> (hermes-style, not XML)
# If output has <function= or <parameter= XML, the qwen3_coder format would
# be needed and our hermes switch is wrong — REPORT and halt.
```

---

## 3. Probe (optional, ~1.5h): Thinking vs Instruct TIR hit-rate

`scripts/probe_qwen3.sbatch` is a Roberts-specific (gpu_h200) array job; on
Perlmutter you'd need to swap the partition/qos lines. The probe data from
Roberts is sufficient for the decision — check `data/results/probe_qwen3_thinking.parquet`
and `data/results/probe_qwen3_instruct.parquet` once the Roberts probe
completes (job 8792195 on Roberts, see §2 bottom of this handoff).

Decision rule:
- **Instruct hit-rate ≥ Thinking − 3 %**: use Instruct (faster: no `<think>`
  overhead, ~half the response length → ~2× trainer speed).
  - Flip `MODEL=Qwen/Qwen3-4B-Instruct-2507` in the sbatch.
- **Instruct underperforms by > 3 %**: use Thinking (plan default).

---

## 4. Submit the smoke

```bash
sbatch scripts/perlmutter/smoke_tir_het.sbatch
# Or with Instruct variant:
# sbatch --export=ALL,MODEL=Qwen/Qwen3-4B-Instruct-2507 scripts/perlmutter/smoke_tir_het.sbatch
```

Topology: 1 trainer node + 5 rollout nodes + 1 xverify node (het groups
0/1/2). Same as the 2026-04-17 baseline run for direct comparison.

Key env-vars (already defaulted correctly in the sbatch):
- `TOTAL_STEPS=10`, `TRAIN_BATCH=128`, `ROLLOUT_N=8` → 1024 rollouts/step.
- `TRAIN_SP=2`, `STALENESS=1`, `PPO_MAX_TOKEN_LEN_PER_GPU=20480`,
  `PYTORCH_ALLOC_CONF=expandable_segments:True`.

---

## 5. What to watch

### 5.1 First 3 minutes after job starts

Locations (relative to repo root):
- `logs/<jobname>_<jobid>.log` — main driver log (stdout).
- `logs/<jobname>_<jobid>.err` — stderr.
- `logs/<jobname>_<jobid>.ray_head.log` — Ray head on trainer node.
- `logs/<jobname>_<jobid>.ray_worker_*.log` — Ray rollout workers (5 files).
- `logs/<jobname>_<jobid>.xverify.log` — xVerify service.

**Critical lines to grep for in the first 3 min:**

```bash
# Ray cluster came up:
grep "READY" logs/<jobname>_<jobid>.log
# Expect "[ray-probe] READY" within 1 min of ray-worker start.

# Ulysses SP is actually active:
grep -i "ulysses\|sequence_parallel\|flash" logs/<jobname>_<jobid>.log | head -5
# Expect references to ulysses_sequence_parallel_size=2 and flash-attn backend
# being monkey-patched in. If SP stays at 1, the switch didn't take effect.

# Trainer actor pinned to correct node:
grep "placed on" logs/<jobname>_<jobid>.log
# Expect: "[FullyAsyncTrainer] placed on <trainer-node-hostname>"
# Should match $SLURM_JOB_NODELIST_HET_GROUP_0. If it says a rollout node,
# the trainer_node resource pin didn't work.

# xverify up:
cat outputs/xverify_endpoints/current.url
# Expect a reachable URL within 3 min.

# vLLM loaded model with flash-attn (not sdpa):
grep -i "attention.*impl\|flash_attention" logs/<jobname>_<jobid>.log
```

### 5.2 Per-step metrics to watch (after step 1 lands)

Log lines to parse (from `fully_async_trainer.py` prints):

```
step:1 - fully_async/rollouter/active_time:<X> - version_time:<Y> - idle_ratio:<Z>
Training Progress:  10%|█ | 1/10 [<step_wall>, <seconds>/it]
```

| Metric | Qwen3.5 baseline (2026-04-17) | Qwen3 target | Red flag if |
|---|---|---|---|
| step wall (`<seconds>/it`) | 1634 s | **≤ 300 s** | ≥ 1200 s |
| `rollouter/active_time` | 303 s | 100-250 s | > 500 s |
| `rollouter/version_time` | 1620 s | ~250-500 s | ~1400 s (no Qwen3 speedup) |
| `rollouter/idle_ratio` | 0.813 | 0.3-0.5 | > 0.8 (trainer still the bottleneck) |
| `step 2 completes` | **no (OOM)** | yes | OOM |
| step 1 reward mean | 0.269 | ≥ 0.20 | < 0.05 (model broken) |
| zero-adv group rate | 22.7 % | ≤ 40 % | > 60 % |

If step wall > 600 s, **stop and diagnose** — likely SP not active or
flash-attn not picked up. See §6.1.

### 5.3 Rollout dump sanity (after step 1 finishes)

```bash
ls outputs/physcode_tir_smoke/<EXPERIMENT>/rollout_dumps/
head -1 outputs/physcode_tir_smoke/<EXPERIMENT>/rollout_dumps/1.jsonl | python3 -c "
import json,sys
d = json.loads(sys.stdin.read())
print('keys:', list(d.keys()))
print('output preview:', d.get('output','')[:500])
print('score:', d.get('score'))"
```

Look for: **well-formed hermes tool-call** in the output:
```
... reasoning ...
<tool_call>
{"name":"python","arguments":{"code":"..."}}
</tool_call>
```

If you see `<function=` or `<parameter=` XML markers, format mismatch — the
hermes switch was wrong for this model variant. Halt and flip
`multi_turn.format` back to `qwen3_coder` in `train_async.sh:303`.

---

## 6. Common failure modes

### 6.1 step wall still > 600 s

Likely Ulysses SP didn't take effect. Checks:

```bash
# Grep launcher logs for "ulysses_sequence_parallel_size":
grep -r "ulysses_sequence_parallel" logs/<jobname>_*.log
# Should say "=2" in the Hydra dump around FullyAsyncTaskRunner config.

# Check FlashAttention is used (not SDPA):
grep -i "attn_impl\|attention_impl" logs/<jobname>_*.log
# Should NOT say "sdpa".

# Check use_remove_padding:
grep "use_remove_padding" logs/<jobname>_*.log
# Expect True.
```

If SP is stuck at 1, the verl build may not support it for this model — try
SP=1 (edit `TRAIN_SP=1` in sbatch) and see if the run still gets a speedup
from just removing SDPA + Qwen3_5 wrap policy.

### 6.2 OOM on step 1 or 2

Despite expandable_segments + PPO_MAX_TOKEN_LEN=20480 + SP=2. Ordered fallbacks:

1. `PPO_MAX_TOKEN_LEN_PER_GPU=16384` — one-level shrink.
2. `TRAIN_SP=4` — only if SP=2 works structurally; SP=4 cuts activations further
   but loses FSDP param-sharding (DP=1 on 4 GPUs). Risk: param memory per GPU doubles.
3. `ACTOR_PARAM_OFFLOAD=True` — adds param offload on top of optimizer offload.
4. Drop rollout to 1 node first (FINDINGS §12 #4) — not an OOM fix but frees
   topology for potentially using the extra nodes as trainer nodes.

### 6.3 Trainer placed on wrong node (`[FullyAsyncTrainer] placed on <rollout-node>`)

The `trainer_node` resource pin didn't propagate. Check:
```bash
grep "trainer_node" logs/<jobname>_*.ray_head.log logs/<jobname>_*.ray_worker_*.log
# Expect: ray head started with --resources='{"trainer_node": 1}'.
grep "cluster_resources\|available resources" logs/<jobname>_*.log
```

If broken: fallback to `feat/trainer-node-pin` branch reverted:
```bash
git revert <SHA of wandb-pin commit>
```
Smoke will still run, just with wandb showing wrong GPU node (telemetry bug, not training bug).

### 6.4 Tool calls malformed in rollout dumps

As noted in §5.3. If output has `<function=` / `<parameter=` XML, flip format
back to `qwen3_coder` in `train_async.sh:303` and re-run.

### 6.5 `staleness_threshold=1` degrades reward

Watch `actor/pg_clipfrac` in wandb. If > 30 % after step 5, staleness is causing
too much IS correction. Set `STALENESS=0` in sbatch and re-run.

### 6.6 NCCL/Slingshot issues (inter-node failures)

Collaborator has seen this before; porting Dr.SCI's NCCL env vars is in FINDINGS
§5 (`NCCL_IB_TIMEOUT=32`, `NCCL_NVLS_ENABLE=1`, etc.), minus `NCCL_IB_HCA=mlx5`
which is Azure-specific. If Ray fails to bootstrap 6 nodes, add these to the
sbatch `--env`.

---

## 7. Rollback menu

| Scenario | Command |
|---|---|
| Everything broken → clean slate | `git checkout main && git -C verl checkout physcode` |
| Qwen3 broken, keep wandb fix | `git checkout feat/trainer-node-pin && git -C verl checkout feat/trainer-node-pin` |
| Wandb fix broken, keep Qwen3 | `git revert <58439d8>` on parent, `git revert <1e3bc96f>` on verl |
| Specific Qwen3 commit bad | `git revert <SHA>` — atomic commits are designed for this |

---

## 8. After smoke passes (post-smoke priorities)

Ranked by impact (FINDINGS §12 tier ordering). Do these as separate single-flag
smokes so any regression is attributable:

1. **Rollout 5 → 1 node** (FINDINGS §12 #18). Edit `#SBATCH -N 5` in het-group 1
   → `-N 1` and `NNODES_ROLLOUT=1`. No wall cost expected (rollout idle 81 % in
   baseline). Frees 4 nodes.
2. **`ROLLOUT_ENFORCE_EAGER=False`** (FINDINGS §12 #19). Qwen3 is vanilla
   transformer, CUDA graphs should work. ~10-15 % rollout speedup.
3. **Disable optimizer offload** (`ACTOR_OPT_OFFLOAD=False`). Free if SP=2 gives
   enough activation headroom. Memory-test first with one step.
4. **Bump `PPO_MAX_TOKEN_LEN_PER_GPU` to 32768** (FINDINGS §12 #15 / retool 18432
   heuristic × 1.6). Only after (3) stable.
5. **Trainer 2 nodes** — only if (1)-(4) together don't hit ≤ 5 min/step.
   Requires `_ray_bringup.sh` loop over trainer workers (FINDINGS §8 "Requires
   `_ray_bringup.sh` edit").

---

## 9. Production run (after all smokes pass)

Edit `scripts/perlmutter/prod_tir_het.sbatch` to match the smoke's env-vars
that worked. Then submit with:

```bash
# Stop conditions (from scripts/perlmutter/checklist.md §"Stop conditions"):
# Stop resubmitting when any 2 of these trigger:
#   1. val/score/mean flat within ±0.5% for ≥3 eval points (≥300 steps)
#   2. actor/zero_advantage_group_fraction > 0.70
#   3. actor/response_length/clip_ratio near 0 AND reward ceiling-bound

sbatch scripts/perlmutter/prod_tir_het.sbatch
```

Budget: 48 h × 60 min ÷ (step-time observed in smoke) = target step count.

- If smoke ≈ 5 min/step: **~570 steps** possible (above the 150-step retool benchmark).
- If smoke ≈ 8 min/step: **~360 steps** (comfortable).
- If smoke ≈ 12 min/step: **~240 steps** (tight but viable).

Per FINDINGS §12.6 compound table.

---

## 10. References

### In-repo

- `perlmutter_debug/FINDINGS.md` — investigation report for the 2026-04-17
  failure; includes workload characterization (§3), Dr.SCI and verl-recipe
  comparisons (§5, §6), and the full speedup catalogue (§12).
- `.claude/plans/qwen3-switch.md` — the execution plan this implements (also
  contains paper-framing language for the Instruct vs Thinking choice).
- `scripts/perlmutter/checklist.md` — existing pre-submit + smoke pass
  criteria; still valid.
- `scripts/perlmutter/README.md` — topology/SLURM conventions on Perlmutter.
- `CLAUDE.md` — environment reference; "Required VeRL Overrides for Qwen3.5-4B"
  section is **obsolete after this switch**; consider removing once Qwen3 is
  stable on main.

### External

- Retool recipe (TIR analog, Qwen2.5): `/home/xs272/scratch/refs/verl-recipe/retool/run_qwen2_7b_dapo.sh`
- DAPO recipe: `/home/xs272/scratch/refs/verl-recipe/dapo/run_dapo_qwen2.5_32b.sh`
- Dr.SCI recipe (same data, Qwen3-4B-Base): `/home/xs272/scratch/refs/Dr.SCI/verl/examples/DR_SCI/final/qwen3_long_DrSCI_adaptive_difficulty_full.sh`

### Branches to know

- Parent repo remote: `https://github.com/xingzhis/phys-reasoner.git`
  - `main` — stable, unchanged.
  - `feat/trainer-node-pin` — wandb GPU-pin (tested Roberts, untested Perlmutter).
  - `switch/qwen3-thinking` — Qwen3 infra switch (this handoff's subject).
- Verl repo remote: `git@github.com:xingzhis/verl.git`
  - `physcode` — stable, unchanged (verl fork's "main").
  - `feat/trainer-node-pin` — wandb pin (1 commit above physcode).
  - `switch/qwen3-thinking` — empty (points to same SHA as feat/trainer-node-pin).

### Saved memory caveats

- Roberts: never submit to `priority`/`priority_gpu`/`gpu_priority` partitions
  (billed separately at high rates). Perlmutter's equivalent is unknown — check
  account rates before running a 48h production job.
- Qwen3.5 CUDA-graph issue (`enforce_eager=True`) is **Qwen3.5-specific**;
  Qwen3 does not need this. Flip `ROLLOUT_ENFORCE_EAGER=False` post-smoke
  for rollout speedup.
