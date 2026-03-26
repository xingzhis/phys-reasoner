# GRPO Training Notes

Research and configuration notes for GRPO training on physics reasoning.
Sources: ScaleRL (arxiv 2510.13786), POLARIS (hkunlp.github.io/blog/2025/Polaris/), VeRL repo (volcengine/verl).

---

## Key lessons from POLARIS (directly uses Qwen3-4B, same architecture)

- **Train at 52K, infer at 90K.** Base context is 32K. Extend to 52K during RL training with YaRN
  (rope_scaling factor=1.5), then at inference time the model generalises to 90K without ever training
  at that length. Training at 90K directly is unnecessary and ~40% more expensive.

  ```json
  "rope_scaling": {"rope_type": "yarn", "factor": 1.5, "attn_factor": 1.0}
  ```

- **G=8 rollouts per prompt** is cost-efficient (POLARIS) vs G=16 (ScaleRL). Stick with 8.

- **Multi-stage temperature curriculum** (for 4B thinking models): 1.4 → 1.45 → 1.5 across stages.
  Higher than ScaleRL's T=1.0 because Qwen3's thinking mode needs more entropy.

- **Dynamic difficulty filtering**: at the end of each stage, drop problems where all 8 rollouts score
  > 0.9 (solved) or all score 0 (too hard). This keeps the active training set in the learning zone.

- **Results:** AIME24 81.2%, AIME25 79.4% from Qwen3-4B with this recipe.

## Key lessons from ScaleRL (arxiv 2510.13786)

- **Use large batches**: 256+ prompts/step minimum. Small batches cause early stagnation on downstream
  benchmarks even when validation accuracy keeps improving.

- **FP32 at the LM head** (both generator and trainer): prevents logprob mismatch between vLLM rollout
  and FSDP training. Silently kills training otherwise. VeRL has a setting for this.

- **Zero-advantage filtering**: drop prompts where all G responses have the same reward (zero gradient).
  Roughly 20-30% of prompts early in training. VeRL supports this via `filter_groups.enable`.

- **Token-level loss aggregation** (`loss_agg_mode=token-mean`) is more stable than sample-level
  (seq-mean-token-mean) for long-CoT.

- **16K is the sweet spot** for compute efficiency; 32K improves the performance ceiling but costs 2×.
  Only worth it once you've exhausted the 16K budget.

---

## Memory budget: Qwen3.5-4B on 4×A100 (80 GB each, 320 GB total)

**Fixed FSDP ZeRO-3 overhead across 4 GPUs:**

| Component | Total | Per GPU |
|---|---|---|
| Model weights bf16 | 8 GB | 2 GB |
| Optimizer states fp32 Adam | 48 GB | 12 GB |
| Gradients bf16 | 8 GB | 2 GB |
| **Training overhead** | **64 GB** | **16 GB** |

Leaves ~64 GB/GPU for activations + KV cache.

**KV cache per token** (Qwen3.5-4B: 36 layers, GQA ~8 KV heads, head_dim=128):
`2 × 8 × 128 × 2 bytes × 36 layers ≈ 144 KB/token`

In our zero-shot run (single A100, `max_model_len=86016`): 60.98 GB KV cache available,
499,488 tokens total, 22.8× max concurrency — so even 90K is feasible on 1 A100 for pure inference.

---

## Scenario analysis: 1 node (4×A100), collocated mode

VeRL collocated mode: rollout (vLLM) and training (FSDP) run sequentially on the same GPUs.
Peak memory is the max of each phase independently, so the table below is safe.

| Budget | KV cache/seq | Fits 1 node? | Constraint |
|---|---|---|---|
| 8K | 1.2 GB | yes, comfortably | throughput |
| 16K | 2.4 GB | yes, comfortably | throughput |
| 32K | 4.7 GB | yes (micro-batch=1) | rollout time |
| 52K (POLARIS) | 7.5 GB | tight but fits | micro-batch=1, grad-ckpt |
| 90K | 11.8 GB | tight (OOM risk at large batch) | reduce batch |

With gradient checkpointing (always on for ≥32K), activation memory at 52K is ~9 GB/GPU —
still fits with 16 GB training overhead leaving 55 GB free.

---

## Scaling with nodes

**Rollout dominates step time at long sequences (~85% at 32K+).** Adding nodes helps in two ways:

1. **Async separation** (gen+train on separate GPU pools): overlap rollout and training.
   VeRL's `fully_async` trainer achieves **2.35–2.67× speedup** (measured on Qwen2.5-7B, 128 GPUs).
   Ceiling from overlap alone: `1 / rollout_fraction ≈ 1.18×`; the rest comes from more rollout GPUs.

2. **More rollout GPUs** (TP or replicas): higher vLLM throughput. Near-linear until weight sync
   (NCCL broadcast of 8 GB every update) starts to dominate. For 4B, this bites at ~4 rollout nodes.

**Does adding nodes scale linearly?** No. Rough practical speedups vs 1-node collocated:

| Config | Nodes | Speedup |
|---|---|---|
| 1 node collocated | 1 | 1× |
| 1 gen + 1 train (async) | 2 | ~1.8× |
| 2 gen + 1 train (async) | 3 | ~2.5× |
| 2 gen + 2 train (async) | 4 | ~2.6× |
| 4 gen + 2 train (async) | 6 | ~2.7× |

Beyond ~6 nodes the weight sync overhead starts eating the gains for a 4B model.

---

## Does training time scale linearly with dataset size?

**Wall time per step is constant** — it doesn't depend on dataset size. Total time = steps × time/step.

With 8,383 problems and `train_batch_size=256`:
- Steps per epoch = 8383 / 256 ≈ 33
- Typical convergence: ~20–30 epochs = 660–990 steps

This is similar total steps to POLARIS's 30K dataset at fewer epochs. The smaller dataset means problems
are seen more often per epoch (overfitting / reward hacking risk), which POLARIS's dynamic difficulty
filtering mitigates. With G=8, each epoch sees 8383 × 8 = 67K rollouts — reasonably diverse.

---

## Recommended configs

| Run | Nodes | Layout | Budget | Est. wall time |
|---|---|---|---|---|
| Ablation / sanity check | 1 | collocated, 4 GPUs | 16K | ~2 days |
| **Main Qwen3.5-4B run** | **3** | **2 rollout + 1 train, async** | **52K + YaRN** | **~2.5 days** |
| Scaling check (9B) | 4 | 2 rollout + 2 train, async | 32K | ~3 days |
| Budget push | 4 | 2 rollout + 2 train, async | 52K + YaRN | ~3.5 days |

Note: 9B needs ~126 GB for model+optimizer+gradients, so requires ≥2 nodes for FSDP even before
activations. For 90K tokens the 9B needs 4 nodes minimum.

---

## VeRL async support (latest repo as of 2026-03-24)

| Mode | Status | Speedup | Notes |
|---|---|---|---|
| Collocated (default) | Stable | 1× | `trainer.nnodes`, `trainer.n_gpus_per_node` |
| One-step off-policy | Stable | ~1.4× | `recipe/` folder |
| **Fully async** | `verl/experimental`, recently merged | **2.35–2.67×** | Separate `trainer.*` + `rollout.*` node configs |

### Fully async config sketch (52K, 2 gen + 1 train)

```bash
python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_batch_size=256 \
    data.max_prompt_length=4096 \
    data.max_response_length=52224 \
    actor_rollout_ref.model.path=Qwen/Qwen3.5-4B \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=256 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=131072 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.rollout.temperature=1.4 \
    actor_rollout_ref.rollout.top_p=1.0 \
    actor_rollout_ref.rollout.top_k=20 \
    actor_rollout_ref.actor.use_rollout_log_probs=True \
    algorithm.use_kl_in_reward=False \
    algorithm.filter_groups.enable=True \
    trainer.nnodes=1 \
    trainer.n_gpus_per_node=4 \
    rollout.nnodes=2 \
    rollout.n_gpus_per_node=4 \
    async_training.staleness_threshold=0.5 \
    async_training.trigger_parameter_sync_step=4 \
    async_training.partial_rollout=True \
    trainer.total_epochs=25
```

### Key parameters to tune

- `async_training.staleness_threshold`: 0 = fully on-policy (slower), 0.5 = allow 50% stale (faster,
  slight off-policy). Start at 0.5, watch for training instability.
- `async_training.trigger_parameter_sync_step`: how many local training steps before syncing weights
  back to rollouter. Higher = less sync overhead, more staleness.
- `rollout.tensor_model_parallel_size`: use TP=2 for 32K, TP=4 for 52K+ (distributes KV cache).
- `actor_rollout_ref.rollout.temperature`: POLARIS uses 1.4–1.5 for Qwen3-4B thinking mode.

---

## POLARIS multi-stage training schedule (4B)

Stage 1 → Stage 2 → Stage 3, with increasing temperature and difficulty filtering between stages:

| Stage | Temp | Action |
|---|---|---|
| 1 | 1.4 | Train on full filtered dataset |
| → | | Filter: remove problems with pass@8 > 0.9 |
| 2 | 1.45 | Continue training on harder subset |
| → | | Filter again |
| 3 | 1.5 | Final stage on hardest problems |

Clip ratio stays below 10% at 52K — healthy signal that PPO is not over-updating.
