# 2026-04-22 Overnight Debug Session — Findings & Fixes

## TL;DR

After ~12 hours of debugging "training reward stuck at 0.69, ppo_kl frozen at 8e-4 across 19+ steps":

**Root cause: bf16 logit precision loss in compute_log_prob.** The softmax → log_prob computation in bf16 (~7 bits mantissa) was rounding tiny per-token policy updates to noise, capping `ppo_kl` at ~8e-4 regardless of LR.

**Fix: enable verl's existing `use_fused_kernels=True` path**, which upcasts logits to FP32 before softmax. This is the ScaleRL recommendation (arxiv:2510.13786). At only 5% of full LR, FP32 already produces 1.85× the `ppo_kl` that bf16 produces at FULL LR — confirming precision was the bottleneck.

**Secondary contribution: ScaleRL-style Zero-Variance Filtering (ZVF)** — added a ~40-line patch to `verl/trainer/ppo/ray_trainer.py:compute_advantage` that masks saturated (zero-variance) groups from the loss DENOMINATOR. Saturated groups already contribute 0 gradient (`adv = score - mean = 0`), but they dilute the per-token loss aggregator. Removing them recovers the correct effective batch size. Works for both sync and async paths via the shared `compute_advantage` function.

## Symptoms

- `critic/score/mean` flat at ~0.65–0.75, fluctuating in noise band
- `actor/ppo_kl` constant at ~8e-4 across **20× LR ramp** (1e-7 → 3e-6) and across data filters [1,7], [2,5], [4,4]
- `actor/pg_clipfrac` at ~0.001 (PPO ratio essentially never strays from 1.0)
- `actor/grad_norm` stable at 0.03–0.05
- `critic/group/saturated_frac` at 0.4–0.6 (50%+ groups saturated)
- Pattern persisted across:
  - 3 different LR values (1e-6, 3e-6 from prior runs)
  - 3 different data filters (broader → tighter)
  - sync vs async trainer (sync DAPO showed same pattern)
  - TIR vs CoT (CoT also flat per yesterday's data)

## What we tried (in order)

1. **Tighter Goldilocks filter [2,5] then [4,4]** — no improvement. Static filtering can't fix per-step saturation drift; tight filter still produced ~50% saturated groups at temp=1.4 stochastic rollouts.

2. **Switch to sync DAPO trainer** — successfully ran, filter_groups (DAPO's batch-level dynamic filter) pushed `informative_frac` to 1.0, but `ppo_kl` STILL frozen at 8e-4. Disproved "filter_groups solves the freeze" hypothesis.

3. **POLARIS-aligned hparams** (LR=1e-6, KL=0.001, asymmetric clip 0.2/0.28) — already aligned; not the issue.

4. **ScaleRL-style ZVF mask in async** — implemented via 40-line patch in `compute_advantage`. Verified mask fires (`[ZVF] kept N/M samples` log line), `grad_norm` 2× larger as the math predicts. But `ppo_kl` STILL frozen — disproved "loss-denominator dilution" as sole cause.

5. **FP32 logits via `use_fused_kernels=True`** — **THIS IS THE FIX.** `ppo_kl` jumped from 8e-4 to 1.5e-3 at only 5% LR (warmup step 1). Per-unit-LR responsiveness ~35× higher.

## Code changes (committed)

### Parent repo
- `scripts/train_async.sh`: added env hooks for `KL_COEF`, `FILTER_GROUPS_*`, and `USE_FUSED_KERNELS` / `FUSED_KERNEL_BACKEND`
- `scripts/serve_xverify_vllm.py` + `scripts/perlmutter/serve_xverify_vllm_40g.sbatch`: vLLM AsyncLLMEngine xverify server (~200 rps vs ~5 rps for transformers serial)
- `scripts/score_and_filter_vllm.py` + `scripts/refilter_from_scored.py`: vLLM-batched offline scoring; refilter from existing scored parquet without re-rolling
- `scripts/monitor_grpo.sh`: live per-step health table (info_frac, ppo_kl, grad_norm, etc.)
- `scripts/train_dapo_sync.sh` + `scripts/perlmutter/prod_tir_dapo_sync_40g.sbatch`: sync DAPO variant for filter_groups testing
- `src/phys_reasoner/tir/sandbox.py`: catch `ValueError` (covers `UnicodeEncodeError`) in `ast.parse` so model-emitted surrogate escapes don't kill rollout workers
- `src/phys_reasoner/tir/prompts.py`: try/except `UnicodeDecodeError` in `extract_tool_call_code` for truncated `\\U` escapes
- `eval/inference/rollout.py`: scrub UTF-16 surrogates from string fields before pyarrow `to_parquet`

### verl submodule (branch: `feature/zvf-mask`, commit 84b889d2)
- `verl/trainer/ppo/ray_trainer.py:compute_advantage` GRPO branch: added ZVF mask. Gated on existing `algorithm.filter_groups.enable` flag (default False = no-op). Works for both sync (`recipe.dapo.main_dapo`) and async (`fully_async_policy.fully_async_main`) paths since both call the same `compute_advantage`. Reference: ScaleRL, arxiv:2510.13786.

## Key configuration discoveries

- **`use_fused_kernels=True` + `fused_kernel_options.impl_backend=torch`** triggers Qwen3's FP32 logit path via `verl/models/transformers/dense_common.py:forward_with_torch_backend` → `verl/utils/experimental/torch_functional.py:_fused_linear_for_ppo_fwd` (line 35: `logits = logits.to(torch.float32)`)
- **Verification at runtime**: log line `Using Torch backend for fused kernels in Qwen3ForCausalLM` (positive); absence of `Skipping monkey patch ... use_fused_kernels is False` (negative)
- **`recipe/dapo/dapo_ray_trainer.py:240-299`**: actual filter_groups implementation. Was missing locally because `verl/recipe/` was an uninitialized git submodule. Run `git submodule update --init recipe` to get it.
- **Async path filter_groups was DEAD CODE**: shell script `dapo_30b_a3b_base_math_fsdp.sh:48-49` declares `enable_filter_groups=True` but never passes it to the python CLI. Confirmed `verl.experimental.fully_async_policy` has no filter_groups consumer.

## Submitted overnight runs (2026-04-22 23:30)

- **51916258 (premium)**: FP32 ZVF TIR on `processed_tir_filtered` (8.8k prompts). Wandb: `physcode_tir/runs/xn0601pu`
- **51917864 (premium)**: FP32 ZVF CoT on `processed_cot_filtered` (8.8k prompts) — for paper TIR-vs-CoT comparison
- **51917866 (regular)**: backup FP32 ZVF TIR on unfiltered 17k — fallback if 107k pipeline doesn't deliver
- **107k scoring** (running separately, ETA ~2h) — once done, can refilter to ~50k prompts for a paper-grade larger run

## Reference
- ScaleRL (Meta, 2025): https://arxiv.org/abs/2510.13786 — FP32 logits + ZVF
- POLARIS-4B (HKU NLP, 2025): https://hkunlp.github.io/blog/2025/Polaris/ — same base model recipe
- DAPO (ByteDance, 2025): https://arxiv.org/abs/2503.14476 — clip_ratio_high=0.28, filter_groups
- Dr.GRPO (sail-sg, 2025): https://arxiv.org/abs/2503.20783 — `norm_adv_by_std_in_grpo=False`
- verl filter_groups: `verl/recipe/dapo/dapo_ray_trainer.py:240-299`
- verl FP32 fused linear: `verl/utils/experimental/torch_functional.py:27-49`
- verl monkey patch: `verl/models/transformers/monkey_patch.py:235-283`
