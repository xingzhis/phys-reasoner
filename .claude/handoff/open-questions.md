# Open questions — resolve on arrival

Each item is unknown to the Roberts-side Claude that prepared this handoff. Resolve by **(a)** grepping a log once the smoke runs or **(b)** asking the user. Don't assume.

## 1. NCCL tuning env vars — keep or drop?

Added to `scripts/perlmutter/smoke_tir_het.sbatch` in commit `2932311`:
```
NCCL_IB_TIMEOUT=32
NCCL_NVLS_ENABLE=1
NCCL_IBEXT_DISABLE=1
TORCH_NCCL_ENABLE_MONITORING=0
```
Lifted from Dr.SCI's launch (Azure IB). Perlmutter is Slingshot. Plausibly fine, untested.

**How to resolve:** compare to the collaborator's most recent working multi-node sbatch on Perlmutter. If they do not set these, ask the user whether to keep or drop.

## 2. Is Ulysses SP actually firing?

`train_async.sh` sets `TRAIN_SP=2` by default. Need to confirm verl monkey-patches flash-attn at model load.

**How to resolve (once smoke launches):** grep launcher log for `Monkey patch _flash_attention_forward` (expected) or `ulysses_sequence_parallel_size=2` in the Hydra dump. If SP stays at 1, the switch didn't take effect — report to user before concluding the step time.

## 3. Is `use_remove_padding=True` firing?

Should see `use_remove_padding: True` in verl's config dump at startup.

**How to resolve:** `grep use_remove_padding logs/<job>.log`. If False, ask user — FSDP memory math changes materially.

## 4. `ROLLOUT_ENFORCE_EAGER=False` safe on Qwen3-Thinking?

The `!!!` degeneration bug was Qwen3.5-specific (B200 TRTLLM + GDN). Qwen3-Thinking is vanilla transformer, should tolerate CUDA graphs. But unverified at scale.

**How to resolve:** after step 1, inspect the rollout dump `outputs/.../rollout_dumps/1.jsonl`. If `!!!` repetitions or truncated-at-same-length outputs appear in >5% of rollouts, flip `ROLLOUT_ENFORCE_EAGER=True` and rerun. Ask user first.

## 5. sbatch account / queue / constraint

`scripts/perlmutter/smoke_tir_het.sbatch` has `TODO(collab)` at lines matching:
- `#SBATCH -A TODO_ACCOUNT_g`
- `#SBATCH -q regular` (and `-q shared` for het-group 2)
- `#SBATCH -C gpu`

**How to resolve:** read collaborator's latest working sbatch. Ask user to confirm the values before committing the resolution.

## 6. Prior Slingshot failures to watch for?

Collaborator may have seen NCCL/IB issues before. Worth asking once: "any known sbatch-time flakiness we should plan around?"
