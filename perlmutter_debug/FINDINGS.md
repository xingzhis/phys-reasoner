# Perlmutter Smoke Investigation — Findings (2026-04-18)

Fact-oriented summary of observations from the 2026-04-17 het smoke run on Perlmutter
(SLURM job 51693591–51693593). Companion files:
- Logs: `perlmutter_debug/logs/`
- Outputs: `perlmutter_debug/outputs/physcode_tir_smoke/smoke_tir_het_20260417.122647/`
- Reference repos cloned to: `/home/xs272/scratch/refs/{Dr.SCI,verl-recipe}`

Recommendations are collected in the last section and marked as such; everything above
that section is observation or direct comparison.

---

## 1. What ran

- **Sbatch file**: `scripts/perlmutter/smoke_tir_het.sbatch` (commit 8582629 or later).
- **Topology (het groups)**:
  - Group 0 (trainer): 1 node × 4 H100/A100 → `HEAD_NODE` (Perlmutter is A100-80G).
  - Group 1 (rollout):  5 nodes × 4 GPUs (20 GPUs total).
  - Group 2 (xverify):  1 node × 1 GPU (shared queue).
- **Model**: `Qwen/Qwen3.5-4B` (hybrid attention: GDN + standard Transformer layers).
- **Framework**: verl experimental `fully_async_policy` path
  (`verl/verl/experimental/fully_async_policy/fully_async_trainer.py`),
  with `actor_rollout_ref.rollout.mode=async` and `hybrid_engine=False`
  (disjoint rollout/trainer GPU pools).
- **Async settings**: `async_training.partial_rollout=False`,
  `async_training.staleness_threshold=0` (i.e., fully synchronous on-policy despite the
  async code path).
- **Smoke budget**: `TOTAL_STEPS=10`, `TRAIN_BATCH=128` (prompts), `ROLLOUT_N=8` →
  1024 rollouts per step.

## 2. What happened (from `logs/physcode_smoke_tir_het_51693591.log`)

| Wall time | Event |
|---|---|
| 19:29:20 | `[FullyAsyncTrainer]` initialized (`use_trainer_do_validate: False`). |
| 19:36:20 | `step:0` — rollouter warmup complete (`version_time:247.67s`, `idle_ratio:1.0`). |
| 19:36:20 → 19:41:23 | Batch-1 collection: **303.03s** (Collected 64/128 at 19:40:08, 128/128 at 19:41:23). |
| 19:41:23 → 20:03:16 | Trainer update for step 1: **1,313s ≈ 21:53**. |
| 20:03:16 | `global_steps: 1`, `param_sync: 4.35s`, `current_param_version: 1`. |
| 20:03:20 | `step:1`, `active_time:303.03`, `version_time:1620.56s`, `idle_ratio:0.8130`. |
| 20:03:24 | `Training Progress: 1/10 [27:14<4:05:08, 1634.31s/it]`. |
| 20:03:24 → 20:07:05 | Batch-2 collection: **220.62s** (vLLM cache warm). |
| ~20:07:05 → 20:29 | Trainer update for step 2 **OOM** during `loss.backward()`. |
| 13:10:30 (UTC) | Ray head raylet killed with `SIGKILL (-9)` — cluster brought down. |

Observed per-step wall: **27:14 (1634 s/it)** for step 1.
Rollout active fraction of step: **18.6 %**; rollouter idle fraction: **81.3 %**.

### OOM details (from step-2 crash trace)
```
torch.OutOfMemoryError: CUDA out of memory.
  Tried to allocate 11.54 GiB.
  GPU 0 has a total capacity of 79.25 GiB of which 11.39 GiB is free.
  Including non-PyTorch memory, this process has 67.84 GiB memory in use.
  Of the allocated memory 49.90 GiB is allocated by PyTorch,
  and 11.72 GiB is reserved by PyTorch but unallocated.
Stack: engine_workers.py:627 update_actor → loss.backward().
```
- Peak occurred on *step 2*, not step 1 (step-1 activation state was slightly smaller).
- `expandable_segments:True` was **not** set in the sbatch `--env` list.
- `actor_rollout_ref.actor.fsdp_config.optimizer_offload=True` was in effect.

### Non-fatal warnings in the log
- `WARNING: Overriding HOME environment variable with APPTAINERENV_HOME is not permitted`
  — apptainer refused to set `HOME=/tmp`; project sets `HF_HOME`, `XDG_CACHE_HOME`,
  `TRITON_CACHE_DIR`, `MPLCONFIGDIR` explicitly so this is harmless but noted.
- `WARNING: Environment variable PYTHONPATH already has value […] will not forward new value […]`
  — parent-shell PYTHONPATH contained collaborator-specific paths
  (`/opt/nersc/pymon`, `DeepGWBSE`, `EXPH`, `BGWAgent`) repeated 3×. Apptainer
  correctly refused; container-internal PYTHONPATH ended up correct
  (`.async-extras:/opt/phys-extras/`).
- `RewardLoopWorker … WARNING … Timeout is disabled` — originates from the TIR
  Python-sandbox executor; rollout-side only, not reward-side. All xverify POSTs in
  `logs/…xverify.log` returned `200 OK` in < 5 s.
- `Using blocking ray.get inside async actor. This blocks the event loop.`
  (one-shot on trainer init; visible in `train.log`.)

### Hydra overrides actually passed (extracted from the error trace)
Captured verbatim from the crash log; notable fields:
```
data.max_prompt_length=1024
data.max_response_length=18961
actor_rollout_ref.rollout.max_model_len=19985
actor_rollout_ref.rollout.max_num_seqs=1024
actor_rollout_ref.rollout.tensor_model_parallel_size=4
actor_rollout_ref.rollout.gpu_memory_utilization=0.85
actor_rollout_ref.rollout.enforce_eager=False
actor_rollout_ref.rollout.multi_turn.format=qwen3_coder
actor_rollout_ref.rollout.multi_turn.thinking_budget=12288
actor_rollout_ref.rollout.multi_turn.tool_call_budget=2048
actor_rollout_ref.rollout.multi_turn.max_assistant_turns=2
actor_rollout_ref.rollout.multi_turn.max_user_turns=1
actor_rollout_ref.rollout.multi_turn.max_parallel_calls=1
actor_rollout_ref.actor.ppo_mini_batch_size=128
actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1
actor_rollout_ref.actor.use_dynamic_bsz=True
actor_rollout_ref.actor.ppo_max_token_len_per_gpu=24576
actor_rollout_ref.actor.fsdp_config.param_offload=False
actor_rollout_ref.actor.fsdp_config.optimizer_offload=True
'+actor_rollout_ref.model.override_config={attn_implementation:sdpa}'
actor_rollout_ref.actor.use_rollout_log_probs=True
actor_rollout_ref.actor.use_kl_loss=False
actor_rollout_ref.actor.kl_loss_coef=0.0
actor_rollout_ref.actor.entropy_coeff=0
actor_rollout_ref.actor.clip_ratio_low=0.2
actor_rollout_ref.actor.clip_ratio_high=0.28
actor_rollout_ref.actor.clip_ratio_c=10.0
actor_rollout_ref.actor.optim.lr=1e-6
actor_rollout_ref.actor.optim.lr_scheduler_type=constant
actor_rollout_ref.actor.optim.lr_warmup_steps=20
actor_rollout_ref.ref.fsdp_config.param_offload=True
rollout.nnodes=5  rollout.n_gpus_per_node=4
trainer.nnodes=1  trainer.n_gpus_per_node=4
async_training.partial_rollout=False
async_training.staleness_threshold=0
async_training.use_trainer_do_validate=False
```

## 3. Workload characterization (length, reward)

### 3.1 Step-1 rollout dump
`perlmutter_debug/outputs/physcode_tir_smoke/smoke_tir_het_20260417.122647/rollout_dumps/1.jsonl`
contains **1,024 rollouts** (128 prompts × 8). Schema keys: `input, output, gts, score,
step, acc`. Tokenized with `Qwen/Qwen3.5-4B` tokenizer.

**Reward (binary 0/1):**
- mean score = **0.269**; 73.1 % score=0, 26.9 % score=1, 0 % other.
- `acc` identical to `score` on this dump.
- Per-prompt mean reward distribution (128 groups of 8):

  | Band | Count | % |
  |---|---|---|
  | 0.00 | 28 | 21.9 % |
  | 0.01–0.25 | 34 | 26.6 % |
  | 0.25–0.50 | 37 | 28.9 % |
  | 0.50–0.75 | 19 | 14.8 % |
  | 0.75–0.99 | 9 | 7.0 % |
  | 1.00 | 1 | 0.8 % |

- **Zero-advantage groups** (group_std < 1e-6): **29/128 = 22.7 %**.

**Token lengths (Qwen3.5-4B tokenizer, full population):**

| | prompt | response | prompt+response |
|---|---|---|---|
| mean | 615 | **6,285** | 6,900 |
| p50 | 594 | **3,872** | 4,476 |
| p75 | 632 | 12,747 | 13,432 |
| p90 | 699 | **14,352** | 14,938 |
| p95 | 766 | 14,353 | 15,037 |
| p99 | 889 | 14,694 | 15,317 |
| max | 949 | 18,958 | 19,844 |

**Response-length tail fractions** (n=1024):
```
> 2k: 74.0%    > 4k: 48.9%    > 6k: 37.4%    > 8k: 33.4%
>10k: 30.9%   >12k: 27.2%   >14k: 12.4%   >16k:  1.0%
>18k:  0.7%
```

**Length conditional on reward:**
- score=0 (n=749): response mean 5,815, p50 3,626, p90 14,353.
- score=1 (n=275): response mean **7,564**, p50 **7,894**, p90 13,314.
- → correct responses average ~30 % longer than wrong ones; their p50 is close to
  double the p50 of wrong responses.

### 3.2 Earlier probe dumps (base-model, same budgets)
From `outputs/probe_v5_A/rollouts.parquet` and `outputs/probe_v5_B/rollouts.parquet`
(10,000 rollouts each, Qwen3.5-4B, `thinking_budget=12288`, `tool_call_budget=2048`,
`answer_budget=4096`, `interrupt_len=17`):

| | probe_A | probe_B |
|---|---|---|
| rows | 10,000 | 10,000 |
| interrupt rate | 2175 / 10,000 = **21.8 %** | 2132 / 10,000 = **21.3 %** |
| total-tokens median | 4,318 | 4,013 |
| total-tokens mean | 6,454 | 6,060 |
| total-tokens p75 | 12,320 | 10,473 |
| total-tokens p90 | 14,336 | 14,336 |
| total-tokens p99 | 15,155 | 14,543 |
| total-tokens max | 19,980 | 24,063 (prompt+resp) |

Per-phase token stats (probe_A, 500 sampled rollouts):

| phase | mean | p50 | p90 | p99 | max |
|---|---|---|---|---|---|
| phase1_text (thinking+assistant) | 4,779 | 1,935 | **12,288** | 12,289 | 12,290 |
| phase1b_text | 398 | 0 | 2,048 | 2,048 | 2,049 |
| code | 472 | 258 | 1,158 | 2,910 | 8,639 |
| sandbox_stdout | 89 | 0 | 244 | 893 | 1,410 |
| phase2_text | 717 | 466 | 1,824 | 4,096 | 4,096 |

`phase1_text` saturates at the 12,288 cap for the tail. `phase2_text` saturates at
the 4,096 cap only rarely (p99 = 4,096).

Interrupt + length breakdown by primary answer type (probe_A, full 10k, char→token
approx factor 3.6):
```
equation     n=3040  mean~4595  p50~3261  p90~11097  p99~15149
expression   n=2176  mean~5289  p50~3544  p90~12152  p99~15200
mcq          n=1920  mean~7357  p50~6820  p90~14083  p99~16724
numerical    n=2488  mean~4636  p50~2937  p90~11660  p99~15573
interval     n=  48  mean~9338  p50~11675 p90~14501  p99~17369
true_false   n=  88  mean~4138  p50~2967  p90~10013  p99~12859
```

### 3.3 xverify

`logs/physcode_smoke_tir_het_51693593.xverify.log`: every POST returned 200 OK within
0.2–5 s; most cluster sub-1 s. Reward scoring across 1024 rollouts completed within
the 303 s rollout phase (not a separate stage), so it is bounded above by 303 s and
in practice much less. Verifier is **not** on the critical path.

## 4. Where the 22-minute step spends its time (inferred)

`logs/physcode_smoke_tir_het_51693591.log` carries only coarse phase markers; no
per-phase `timing_s/*` telemetry was emitted (the trainer-side wandb logger was not
yet in place — the user is currently fixing that). The decomposition is therefore
inferred from config + empirical length distribution, not directly measured.

- `ppo_max_token_len_per_gpu = 24576` → packed-token micro-batch ≤ 24,576 tokens per GPU.
- `data.max_prompt_length + data.max_response_length = 1024 + 18961 = 19,985` tokens →
  at most 1 full-length sequence per packed micro-batch on a GPU, more if the batch
  is short-tailed.
- 1024 rollouts / 4 GPU = 256 rollouts/GPU. With avg length ~6.9k (step-1 dump),
  packed dimension gives ~3–4 sequences per micro-batch → **~70–85 micro-batches**
  per pass through the mini-batch.
- Passes per step: ref-log-prob (1 fwd, param-offload CPU↔GPU), actor fwd+bwd (1 each)
  → effectively 3 full passes.
- Optimizer offload: AdamW state for ~4.54 B params is ~36 GB total (fp32, 2 moments),
  sharded 4-way → ~9 GB/GPU. With `optimizer_offload=True` this state lives on CPU
  and crosses PCIe each step.
- Under these packing + offload conditions, community reports on A100 class show
  ~5–8 s per gradient-checkpointed fwd+bwd step for a 4B model at ~6k average tokens.
  80 micro-batches × 3 passes × ~5 s ≈ 20 min. Matches the observed 22 min ± smoke
  overhead.

## 5. Reference comparison A: Dr.SCI (same data, different model)

Repo: <https://github.com/MiniByte-666/Dr.SCI> (cloned to `/home/xs272/scratch/refs/Dr.SCI`).
Trains Qwen3-4B-Base on the Dr.SCI corpus (same base data family as this project uses).
Script: `verl/examples/DR_SCI/final/qwen3_long_DrSCI_adaptive_difficulty_full.sh`.

| Item | Dr.SCI | This project |
|---|---|---|
| Model | Qwen3-4B-Base | **Qwen3.5-4B** |
| Attention backend | `VLLM_ATTENTION_BACKEND=FLASH_ATTN`; flash-attn throughout | `attn_implementation: sdpa` (forced; see CLAUDE.md note) |
| Framework | Standard verl PPO, hybrid engine | `fully_async_policy` experimental, disjoint pools |
| FSDP world size | 8 per node (TP=1 rollout) | 4 (1 trainer node) |
| `actor.fsdp_config.param_offload` | False | False |
| `actor.fsdp_config.optimizer_offload` | **False** | **True** |
| `ref.fsdp_config.param_offload` | True | True |
| `ppo_max_token_len_per_gpu` | **32,768** | 24,576 |
| `ppo_mini_batch_size` | 256 | 128 |
| `ppo_micro_batch_size_per_gpu` | 2 | 1 |
| `rollout.tensor_model_parallel_size` | 1 | 4 |
| `rollout.gpu_memory_utilization` | 0.55 | 0.85 |
| `rollout.max_num_batched_tokens` | 32,768 (explicit) | default |
| `max_prompt_length` | 2,048 | 1,024 |
| `max_response_length` | 14,336 | **18,961** |
| `data.max_validation_length` | 30,720 | (not set) |
| `thinking_budget` | (not used; hard cap at max_response_length) | 12,288 |
| `use_kl_loss` / `kl_loss_coef` / `kl_loss_type` | **True / 0.001 / low_var_kl** | False / 0 / – |
| LR schedule | cosine, warmup 10 | constant, warmup 20 |
| LR | 1e-6 | 1e-6 |
| NCCL env | `NCCL_IB_HCA=mlx5`, `UCX_NET_DEVICES=mlx5_*`, `NCCL_IB_TIMEOUT=32`, `NCCL_NVLS_ENABLE=1`, `NCCL_IBEXT_DISABLE=1`, `TORCH_NCCL_ENABLE_MONITORING=0` | (not set) |
| Reward | `rubric_verifier` (Qwen3-4B GenRM) + answer check | xVerify-7B via HTTP + rule verifier |
| Ulysses SP | (default 1; can use because Qwen3 is standard transformer) | 1 (forced) |
| Chat template | custom `examples/chat_templates/Qwen3_self.jinja` | default Qwen3.5 template |

Reported reproducibility (README table): Qwen3-4B-Base + Dr.SCI (their full pipeline) →
GPQA-D 62.7 (vs Qwen3-4B thinking 55.9), HLE 5.86 (vs 4.52). So their setup produces
real gains on the same base-data family.

## 6. Reference comparison B: verl-recipe

Repo: <https://github.com/verl-project/verl-recipe> (cloned to
`/home/xs272/scratch/refs/verl-recipe`). Three relevant recipes examined.

### 6.1 retool (`retool/run_qwen2_7b_dapo.sh`)
Paper: ReTool — RL for Strategic Tool Use in LLMs (arXiv 2504.11536). Direct TIR
analog with Qwen2.5-7B (SFT → GRPO) on DAPO-Math-17k.

| Item | Value |
|---|---|
| Model | `multiturn-sft-qwen-2.5-7b-instruct` (after retool SFT) |
| Framework | Standard verl, `rollout.mode=async` with hybrid engine |
| `ulysses_sequence_parallel_size` | **4** |
| `fsdp_config.{param,optimizer}_offload` | True, True |
| `ppo_mini_batch_size` | 16 |
| `train_batch_size` (prompts) | 64 |
| `rollout.n` | 16 |
| `actor_max_token_len_per_gpu` | `(max_prompt + max_response) * 1 = 18,432` |
| `log_prob_max_token_len_per_gpu` | `actor * 4 = 73,728` |
| `rollout.tensor_model_parallel_size` | 4 |
| `rollout.gpu_memory_utilization` | 0.9 |
| `max_prompt_length` | 2,048 |
| `max_response_length` | 16,384 |
| `max_turns` (assistant/user) | 16 / 16 |
| `multi_turn.format` | hermes |
| `clip_ratio_low / high / c` | 0.2 / 0.28 / 10.0 |
| `use_kl_loss`, `kl_loss_coef` | False, 0 |
| `use_dynamic_bsz` | True |
| `use_remove_padding` | True |
| `enable_gradient_checkpointing` | True |
| Hardware | `trainer.n_gpus_per_node=8, nnodes=1` |
| LR | 1e-6 |
| Results after 150 GRPO steps (README) | AIME-2025 `acc/mean@30 = 0.60`; `num_turns/mean = 10` |
| Results after 250 PPO steps | AIME-2025 `0.55`; `num_turns/mean = 8.3` |

### 6.2 dapo (`dapo/run_dapo_qwen2.5_32b.sh`)
Paper: DAPO (arXiv 2503.14476). Reproduces 52 % on AIME-2024 with Qwen2.5-32B.

| Item | Value |
|---|---|
| `ulysses_sequence_parallel_size (sp_size)` | **8** |
| `fsdp_config.{param,optimizer}_offload` | True, True |
| `ref.fsdp_config.param_offload` | True |
| `ref.ulysses_sequence_parallel_size` | 8 |
| `actor_ppo_max_token_len` | `max_prompt + max_response` = 2048 + 20480 = 22,528 |
| `infer_ppo_max_token_len` | same as actor |
| `ref.log_prob_max_token_len_per_gpu` | same as actor |
| `rollout.log_prob_use_dynamic_bsz`, `ref.log_prob_use_dynamic_bsz` | True, True |
| `rollout.max_num_batched_tokens` | `max_prompt + max_response` (explicit) |
| `rollout.enable_chunked_prefill` | True |
| `rollout.gpu_memory_utilization` | 0.80 |
| `rollout.tensor_model_parallel_size (gen_tp)` | 4 |
| `use_kl_loss`, `kl_loss_coef` | False, 0 |
| `clip_ratio_low / high / c` | 0.2 / 0.28 / 10.0 |
| `entropy_coeff` | 0 |
| `loss_agg_mode` | token-mean |
| `grad_clip` | 1.0 |
| Algorithmic: `algorithm.filter_groups.enable` | **True** (see §6.3) |
| Reward shaping: `reward_model.overlong_buffer.enable` | **True** (see §6.3) |
| `overlong_buffer.len` | 4,096 |
| `overlong_buffer.penalty_factor` | 1.0 |
| `reward_manager` | dapo (not custom) |
| `trainer.{n_gpus_per_node, nnodes}` | 8, 16 |
| LR | 1e-6, warmup 10, weight_decay 0.1 |
| `val_top_p` | 0.7 |
| `top_k` (rollout) | -1 (vLLM) |
| Hardware of published run | 16 × 8 × H800, image `hiyouga/verl:ngc-th2.6.0-cu126-vllm0.8.3-flashinfer0.2.2-cxx11abi0` |

### 6.3 DAPO features called out by the paper/recipe
1. **Decoupled clip ratios (a.k.a. Clip-Higher)** — asymmetric PPO clip:
   ```python
   pg_losses1 = -advantages * ratio
   pg_losses2 = -advantages * torch.clamp(ratio, 1 - cliprange_low, 1 + cliprange_high)
   pg_losses  = torch.maximum(pg_losses1, pg_losses2)
   ```
   `clip_ratio_low=0.2`, `clip_ratio_high=0.28`. (Your config already uses these.)

2. **Dynamic sampling with group filtering** — `algorithm.filter_groups.enable=True`,
   `filter_groups_metric=acc`, `max_num_gen_batches=10`,
   `data.gen_batch_size = train_prompt_bsz * 3`. Over-generates rollout groups; any
   group whose `acc` metric is all-same is dropped and another generated; effective
   batch always has non-zero group advantage. Implemented in
   `recipe/dapo/main_dapo.py` (separate entrypoint).

3. **Overlong reward buffer** — `reward_model.overlong_buffer.enable=True`,
   `overlong_buffer.len=4096`, `overlong_buffer.penalty_factor=1.0`. Soft reward
   penalty applied to the last `overlong_buffer.len` tokens of the response; length
   pressure expressed through reward rather than hard truncation. Implemented in the
   `dapo` reward manager.

4. **Token-level PG loss** — `loss_agg_mode="token-mean"` (not sample-mean); same as
   your config.

### 6.4 r1 (`r1/run_r1_distill_qwen.sh`)
Distillation-style GRPO on DeepSeek-R1 teacher outputs. Less directly relevant —
not TIR, not scientific reasoning. Noted for completeness.

## 7. Structural constraints specific to Qwen3.5

Evidence (from this project's CLAUDE.md and repo's README notes):
- `transformers==5.3.0` + `flash-linear-attention==0.4.2` required because Qwen3.5
  has hybrid Gated Linear Attention (GDN) layers; installed in `/opt/phys-extras/`
  overlay (not in SIF).
- `override_config={attn_implementation:sdpa}` required because verl's Ulysses
  monkey-patch on flash-attn is incompatible with GDN; symptom "CUDA illegal memory
  access". SDPA bypasses the monkey-patch.
- `Qwen3_5ForCausalLM._no_split_modules` incorrectly lists `Qwen3_5VisionBlock`
  (absent from text-only model); explicit wrap policy override required:
  `+actor_rollout_ref.actor.fsdp_config.wrap_policy.transformer_layer_cls_to_wrap=[Qwen3_5DecoderLayer]`.

Consequences (stated as facts about the stack, not opinion):
- **No Ulysses SP**: cannot set `ulysses_sequence_parallel_size > 1` without
  re-introducing the flash-attn Ulysses monkey-patch that crashes on GDN. Every
  community long-context recipe examined (retool SP=4, dapo SP=8) uses SP.
- **SDPA vs FlashAttention**: on A100 with ≥ 14 k sequences, SDPA per-layer
  attention compute and memory are substantially higher than flash-attn. Published
  benchmarks (not run on this cluster; cited from HF / flash-attn docs) commonly
  show ~2× fwd speedup and larger bwd/memory gains for flash-attn at these lengths.
- Dr.SCI uses FLASH_ATTN on Qwen3 (no GDN); verl-recipe retool/dapo same on
  Qwen2.5/Qwen3-base.

## 8. Levers currently exposed in this project's code

Enumerated to show what is an env-var or Hydra flip vs. what is a code change.
Source: `scripts/train_async.sh`, `scripts/perlmutter/smoke_tir_het.sbatch`.

**Pure env/Hydra (no code change, no branch needed except to bundle):**
- `PYTORCH_ALLOC_CONF=expandable_segments:True` — add to sbatch `--env`.
- `PPO_MAX_TOKEN_LEN_PER_GPU` — env-var knob in `train_async.sh` (`:-24576`).
- `THINKING_BUDGET` — env-var knob.
- `TOOL_CALL_BUDGET` / `ANSWER_BUDGET` — env-var knobs.
- `NNODES_ROLLOUT` / `N_GPUS_ROLLOUT` — env-var knobs (must also match `#SBATCH -N`).
- `NNODES_TRAIN` / `N_GPUS_TRAIN` — env-var knobs (must also match `#SBATCH -N`).
- `ACTOR_OPT_OFFLOAD`, `ACTOR_PARAM_OFFLOAD` — env-var knobs.
- `VLLM_GPU_MEM_UTIL` — env-var knob.
- `USE_DYNAMIC_BSZ` — env-var knob.
- `ROLLOUT_ENFORCE_EAGER` — env-var knob (memory note: enforce_eager=True avoids
  CUDA graph degeneration on Qwen3.5 per the saved feedback memory
  `feedback_enforce_eager_degeneration.md`; this smoke ran with it=False).
- `async_training.staleness_threshold` — Hydra (hardcoded `=0` at
  `train_async.sh:315`; trivial to env-var expose).
- `async_training.partial_rollout` — Hydra (hardcoded `=False` at :316).
- `rollout.max_num_batched_tokens` — currently default in Hydra; not env-var.
- `ref.log_prob_use_dynamic_bsz`, `rollout.log_prob_use_dynamic_bsz` — not
  currently set; default probably True in recent verl but worth confirming.
- `ref.log_prob_max_token_len_per_gpu` — currently not set explicitly.

**Requires sbatch edit (simple):**
- `#SBATCH -N` / `--gpus-per-node` lines in trainer or rollout het group.
- Adding NCCL tuning env vars.

**Requires `_ray_bringup.sh` edit (moderate):**
- Multi-trainer-node support: currently Ray is started on one HEAD_NODE only; loop
  over a `TRAIN_WORKER_NODES` array analogously to `WORKER_NODES`.

**Requires new code or upstream patch (high):**
- Enable Ulysses SP on Qwen3.5 — would require fixing verl's Ulysses monkey-patch
  for GDN layers, or migrating model family.
- `filter_groups` under `fully_async_policy` — `algorithm.filter_groups.enable`
  exists in DAPO recipe; whether `verl/experimental/fully_async_policy/` honors it
  needs a grep and likely code integration (DAPO has its own entrypoint
  `recipe/dapo/main_dapo.py`).
- `overlong_buffer` reward shaping — DAPO uses `reward_manager=dapo`; this project
  uses `custom_reward_manager` via `src/phys_reasoner/training/reward.py`. Would
  need to port the buffer math into `compute_score`.

## 9. Points of uncertainty (things we did not measure or confirm)

- Exact breakdown of the 22-min trainer step into ref-log-prob vs actor-fwd vs
  actor-bwd vs optimizer. The trainer-side wandb logger is not yet emitting
  `timing_s/*` — collaborator is working on this.
- Whether `fully_async_policy` supports `algorithm.filter_groups`. Not yet grepped.
- Whether `partial_rollout=True` + staleness > 0 is stable with the current
  `bypass_mode=True` rollout correction setting. Not tested.
- Whether `ref.log_prob_use_dynamic_bsz` defaults to True or False in the verl
  version pinned by this project. Worth one grep.
- Absolute step-time for the Dr.SCI and retool published runs — their shell
  scripts don't log wall-clock per step; can only be read off W&B (retool: 150
  steps; dapo 16×8×H800 run: step times not documented in the recipe).
- Whether the overlay on Perlmutter is `phys-reasoner-overlay-017b.img` (per
  CLAUDE.md) or something else (local version on this machine is `-017.img`).
- Whether collaborator's Perlmutter env has any additional differences vs this
  machine's Roberts env beyond the ones visible in logs.

## 10. Observation-only summary (no recommendations)

- One step completed in **27:14**, step 2 OOM'd in backward.
- Rollout is idle **81 %** of step time in the current sync-on-policy setup.
- Thinking-phase tokens saturate the 12,288 budget for the top-quartile of
  rollouts, accounting for ~22 % interrupt rate in both probes and step-1 dump.
- Correct rollouts are ~30 % longer on average than wrong rollouts.
- 23 % of step-1 prompt groups were zero-advantage (no gradient signal).
- Verifier (xverify) is not on the critical path.
- The two comparable RL-for-reasoning codebases (Dr.SCI, verl-recipe
  retool+dapo) all use Ulysses SP=4 or 8, which is structurally unavailable on
  Qwen3.5 with the current verl stack.
- The packed-token budget `ppo_max_token_len_per_gpu` in community recipes
  targets exactly `max_prompt + max_response` (≈ 20 k in our case); this run
  used 24,576.
- Optimizer offload is used by retool/dapo (with SP to compensate) and *not*
  used by Dr.SCI (with 8-way FSDP, no offload needed). This run has offload on
  with 4-way FSDP.
- DAPO-specific features (`filter_groups`, `overlong_buffer`) are not currently
  wired into this project.

---

## 11. Recommendations (explicitly separated)

These are *not* facts — they are the candidate levers extracted from the above,
in rough order of confidence × impact. The user will make the decisions.

### 11.1 Low-risk, single-sbatch changes
1. `PYTORCH_ALLOC_CONF=expandable_segments:True` — reclaim the 11.7 GB fragmentation
   headroom seen in the OOM trace.
2. `PPO_MAX_TOKEN_LEN_PER_GPU` 24,576 → ~20,480 (principled from
   `max_prompt + max_response`; matches retool/dapo heuristic).
3. `THINKING_BUDGET` 12,288 → ~10,240 (data-motivated; see §3: median-correct
   response is 7,894 tokens, so 10 k preserves more of the correct-response
   distribution than 8 k while still cutting ~17 % activation memory).
4. Shrink rollout to 1 node (verified idle 81 % in §2).
5. Use the curated ≤ 15 k subset already referenced in the data pipeline plan;
   exclude the 29 zero-advantage prompt_ids from §3 as a sanity filter.
6. Add `rollout.max_num_batched_tokens` explicitly (DAPO: `max_prompt + max_response`).
7. Set `ref.log_prob_max_token_len_per_gpu ≈ actor * 4` (retool heuristic) and
   confirm `use_dynamic_bsz` on ref/rollout log_prob paths.

### 11.2 Medium-risk (requires `_ray_bringup.sh` + NCCL tuning + smoke)
8. Trainer 1 node → 2 nodes (8-way FSDP), then flip `optimizer_offload=False`.
   Dr.SCI runs no-offload at FSDP-8, suggesting it fits; would recover most of
   the offload-PCIe tax. New failure modes: inter-node NCCL on Slingshot.

### 11.3 Higher-risk (code or framework change)
9. Enable `async_training.staleness_threshold=1` — ~12 % wall-clock savings if
   the async code path handles it cleanly under `bypass_mode=True`; needs a
   stability check against reward curve.
10. Investigate whether `fully_async_policy` supports `algorithm.filter_groups`.
    If yes, enabling it automates the zero-advantage drop (§3: 23 % of groups).
11. Port DAPO `overlong_buffer` math into `src/phys_reasoner/training/reward.py`
    `compute_score`. Allows raising `max_response_length` without thinking-cap
    artifacts.
12. Model migration Qwen3.5 → Qwen3-4B (or Qwen3-4B-Thinking) — unlocks
    flash-attn + Ulysses SP, matching the Dr.SCI / retool / dapo paths.
    Not for the 2-day window.

### 11.4 Branching strategy suggested
- `smoke/stable` branch: bundle 11.1 items (1–7) with atomic commits.
- `smoke/multi-node` branch: off `smoke/stable`, adds item 8 only.
- Everything else: keep out of the 2-day window.

---

## 12. Full speedup lever catalogue (ranked by estimated impact)

This section enumerates every performance-relevant change discussed or implied by
the comparisons above, ranked by expected **per-step wall-clock impact on the
trainer** (except where labeled otherwise). "Speedup" is multiplicative vs. the
current 27-minute step. Estimates are coarse — sources are noted per row.

Categories:
- **TRAIN-TIME** — reduces the ~22 min trainer phase.
- **WALL-CLOCK** — reduces observed step wall but not compute (e.g. overlap).
- **MEMORY** — prevents OOM but doesn't speed; enables other levers.
- **DATA-EFFICIENCY** — fewer steps needed for same reward (not raw wall).
- **ROLLOUT** — affects the rollout 5-min phase (low leverage: idle 81 %).

### 12.1 Tier-A: 1.5× or greater (structural / multi-node / model switch)

| # | Change | Est. speedup | Category | Effort | Evidence & caveats |
|---|---|---|---|---|---|
| 1 | **Qwen3.5 → Qwen3-4B-Base or Qwen3-4B-Thinking.** Unlocks FlashAttention (direct), Ulysses SP (direct), and all the battle-tested verl codepaths. | **2–3× trainer (cumulative)** | TRAIN-TIME | **High**: SFT cold-start needed; rewires CLAUDE.md assumptions; Qwen3_5DecoderLayer wrap-policy + transformers 5.3.0 overlay become moot. | Dr.SCI (Qwen3-4B-Base) and retool/dapo (Qwen2.5) both run on this path. Flash-attn vs SDPA at 14k+ context: ~1.5–2× (HF flash-attn docs, vllm benchmarks). Ulysses SP=4 on top: additional ~1.3–1.5× on activation-bound backward (DAPO paper §5). These compound. You also inherit a more stable `recipe/dapo` path without `fully_async_policy` experimental risk. |
| 2 | **Trainer 1 node → 2 nodes AND disable `optimizer_offload`**. Goes together: FSDP-8 makes the optim state small enough per-GPU (~4 GB) to stay on-device. | **1.8–2.5× trainer (combined)** | TRAIN-TIME | Medium: edit `_ray_bringup.sh` to start Ray on trainer-worker-1, `#SBATCH -N 2` in het-group 0, NCCL tuning. | Dr.SCI runs no-offload at FSDP-8 on the same model size (§5). CPU-offloaded AdamW with PCIe round-trips is ~1.3–1.5× slower than on-GPU (community reports, verl issues). FSDP-8 vs FSDP-4 is roughly linear on the compute side but ~1.6–1.9× wall due to more inter-node comms. Risk: NCCL over Slingshot on Perlmutter. |
| 3 | **Enable Ulysses sequence parallelism** (`ulysses_sequence_parallel_size=4`). Sharded attention along sequence dim; activation memory drops ~linearly. | **1.3–1.5× trainer** | TRAIN-TIME | Blocked on Qwen3.5 (CLAUDE.md: GDN layers crash verl's flash-attn monkey-patch). Only usable *after* #1 (Qwen3 migration). | retool uses SP=4, dapo uses SP=8. DAPO paper notes SP is essential at 20k+ context. Not independently orderable from #1. |
| 4 | **FlashAttention backend** (`VLLM_ATTENTION_BACKEND=FLASH_ATTN` + `attn_implementation=flash_attention_2`). | **1.5–2× trainer fwd/bwd** | TRAIN-TIME | Blocked on Qwen3.5 (same reason as #3). Part of #1 migration. | See #1 source row. |

### 12.2 Tier-B: 1.1–1.5× (single-sbatch / single-flag changes)

| # | Change | Est. speedup | Category | Effort | Evidence & caveats |
|---|---|---|---|---|---|
| 5 | **`thinking_budget` 12,288 → 8,192** (aggressive). | **1.25–1.35× step** | TRAIN-TIME | Low (env var). | §3 probe data: p90 thinking saturates at 12,288; dropping to 8k cuts ~30 % of activation tokens in the tail. *Risk:* §3.1 shows correct responses have p50 = 7,894, so ~half of correct responses get truncated. Interrupt rate estimated 22 % → ~35 %. May reduce reward gain per step. |
| 6 | **`thinking_budget` 12,288 → 10,240** (conservative). | **1.10–1.17× step** | TRAIN-TIME | Low (env var). | Same mechanism as #5 but smaller cut. Interrupt rate → ~28 %. Preserves ~75 % of correct-response tail vs 50 % for 8k. |
| 7 | **Trainer 1 → 2 nodes alone** (keep `optimizer_offload=True`). | **1.5–1.8× trainer** | TRAIN-TIME | Medium (same as #2 minus the offload flip). | FSDP-8 reduces per-GPU param+grad+activation share. Useful if #2's no-offload flip fails memory-wise. |
| 8 | **Switch `partial_rollout=True` + `staleness_threshold=1`**. Overlaps rollout of step t+1 with trainer update of step t. | **1.10–1.15× wall** | WALL-CLOCK | Low (two Hydra flags — see `train_async.sh:315-316`). | Saves ~3 of the 25 min (rollout no longer sequential). Rollout was idle 81 % → this is where those minutes go. Quality risk: adds off-policy drift; mitigated by `use_rollout_log_probs=True` + DAPO clip. Not tested under `bypass_mode=True` rollout-correction path. |
| 9 | **Disable `optimizer_offload` alone** (current 1 node × 4 GPU). | **1.2–1.4× trainer if fits** | TRAIN-TIME | Low (env var). | Dangerous on current FSDP-4: CLAUDE.md math says 4B × fp32 × 2 moments = 36 GB, sharded 4-way = 9 GB/GPU. Already OOM'd *with* offload on, so this likely OOMs *without* offload unless combined with #7. Listed here for completeness. |
| 10 | **Larger `ref.log_prob_max_token_len_per_gpu` (4× actor)** + `ref.log_prob_use_dynamic_bsz=True`, `rollout.log_prob_use_dynamic_bsz=True`. | **1.10–1.20× step** | TRAIN-TIME | Low (Hydra). | retool: `log_prob_max_token_len_per_gpu = actor * 4`; dapo: same dynamic_bsz on all 3 log-prob paths. Ref pass has no backward → can pack 4× more. The ref log-prob is ~1/3 of trainer-step compute, so ~3× speedup on 1/3 ≈ +10–20 % overall. |
| 11 | **FSDP2 instead of FSDP1** (`fsdp_config.strategy=fsdp2`). | **1.10–1.20× trainer** | TRAIN-TIME | Low-medium (flag flip; untested on Qwen3.5 + GDN). | verl issues + PyTorch 2.4 release notes report FSDP2 is ~10–20 % faster than FSDP1. Risk: unknown interaction with GDN wrap policy. |
| 12 | **`max_response_length` 18,961 → 14,336** (hard cap, Dr.SCI style). | **1.10–1.15× step** | TRAIN-TIME | Low (env). | Alternative to #5/#6; trims the *packed-token* ceiling directly. §3 shows only ~1 % of responses exceed 16k, so the real-world tail is already at ~14k. |
| 13 | **DAPO `overlong_buffer` soft penalty** in reward. | **~1.05–1.10× step** + quality | TRAIN-TIME + QUALITY | Medium (port math into `src/phys_reasoner/training/reward.py::compute_score`; DAPO uses its own `reward_manager=dapo`). | Encourages shorter responses via reward shaping; replaces hard thinking-cap truncation. Slows the response-length drift seen in long RL runs. |

### 12.3 Tier-C: marginal or orthogonal (still worth landing — cheap, stack)

| # | Change | Est. impact | Category | Effort | Evidence & caveats |
|---|---|---|---|---|---|
| 14 | **`PYTORCH_ALLOC_CONF=expandable_segments:True`**. | Reclaims ~11.7 GB fragmentation | MEMORY | Near-zero (sbatch `--env`). | Directly from OOM trace: "11.72 GB reserved by PyTorch but unallocated". Enables tier-B levers to survive. |
| 15 | **`PPO_MAX_TOKEN_LEN_PER_GPU` 24,576 → 20,480**. | Memory safety; ≤5 % slower | MEMORY | Near-zero (env). | Matches retool/dapo principle: `max_prompt + max_response`. 20,480 = 1024 + 19,456 ≈ our max_seq. Small increase in micro-batch count. |
| 16 | **`rollout.max_num_batched_tokens` explicit = max_prompt+max_response**. | Rollout throughput tuning | ROLLOUT | Near-zero. | dapo sets this. Rollout is already fast enough (not the bottleneck). |
| 17 | **NCCL tuning env vars from Dr.SCI**: `NCCL_IB_TIMEOUT=32`, `NCCL_NVLS_ENABLE=1`, `NCCL_IBEXT_DISABLE=1`, `TORCH_NCCL_ENABLE_MONITORING=0`. | Stability, small speed | WALL-CLOCK | Near-zero. | Dr.SCI uses these on AzureML IB fabric. Perlmutter is Slingshot (not mlx5), so `NCCL_IB_HCA=mlx5` is not portable. Others generic. |
| 18 | **Shrink rollout 5 → 1 node**. | 0× speed, −16 GPUs of billable waste | ROLLOUT | Near-zero (sbatch). | idle_ratio = 0.81 confirms headroom. Compute saved, not step-time saved. |
| 19 | **`enforce_eager=True`** on rollout. | Correctness, not speed | ROLLOUT | Near-zero. | Saved memory `feedback_enforce_eager_degeneration.md` explicitly flags this to avoid `!!!` repetition on Qwen3.5. Current smoke had it False — real risk to rollout quality. |
| 20 | **Reduce `ppo_mini_batch_size` 128 → 32 or 16** (retool uses 16, dapo 32). | Neutral-to-slightly-slower, but smaller peak | MEMORY | Low. | More frequent optim steps, not less compute. Helps peak memory. |

### 12.4 Tier-D: data-efficiency (fewer steps needed for same reward)

| # | Change | Est. impact | Category | Effort | Evidence & caveats |
|---|---|---|---|---|---|
| 21 | **DAPO `filter_groups` dynamic sampling**. Over-generates rollout groups; drops all-same groups; refills. | **~1.3× effective reward/step** (removes 23 % zero-advantage groups per §3.1) | DATA-EFFICIENCY | Medium-high: lives in `recipe/dapo/main_dapo.py`, a separate entrypoint. Unknown whether `fully_async_policy` can call it. Needs a grep. | §3.1 zero-adv rate 22.7 % ≈ wasted. filter_groups replaces those with informative groups. Extra rollout cost; given idle_ratio 0.81 this is ~free. |
| 22 | **Static zero-adv prompt filter from step-1 dump**. Drop the 29 prompt_ids with group_std=0. | **~1.23× effective** (bounded by 23 %) | DATA-EFFICIENCY | Near-zero (data filter). | Only 128 prompts observed → not the whole pool. Conservative pre-filter. Inferior to #21 because zero-adv moves as policy improves. |
| 23 | **Goldilocks-filtered ≤ 15 k subset** (your data pipeline plan). | Fewer epochs needed + higher signal density | DATA-EFFICIENCY | Medium (pipeline already designed; see `.claude/plans/data-pipeline.md`). | Full 108k pool is too big for the 2-day step budget anyway. |
| 24 | **Dr.SCI-style Dynamic Difficulty Curriculum (DDC)**. Tracks per-sample accuracy, swaps saturated prompts each epoch. | Small-to-moderate long-run | DATA-EFFICIENCY | High (their impl in `verl/verl/utils/dataset/difficulty_tracker.py` + `dynamic_difficulty_dataset.py`). | Only meaningful past ~100 steps when the policy has moved. Not 2-day window. |
| 25 | **Exploration-Expanding SFT (EESFT)** before RL. | Improves exploration, reduces zero-adv | DATA-EFFICIENCY / QUALITY | Very high (Dr.SCI uses 1M tokens distilled from Qwen3-235B for this). | Not applicable: you skip SFT per CLAUDE.md plan (base model hit rate ≥ 15 %, measured at 26.9 % in §3.1). |

### 12.5 Tier-E: other framework-level changes (explicit for completeness)

| # | Change | Est. impact | Category | Effort | Evidence & caveats |
|---|---|---|---|---|---|
| 26 | **Migrate off `fully_async_policy` → standard verl sync PPO**. Dr.SCI, retool, dapo all use the sync entrypoint. | Uncertain; removes experimental-path overhead. Maybe ~1.1× if async machinery is wasting cycles, maybe 0 otherwise. | TRAIN-TIME | High: different entrypoint, loses async infrastructure. Dr.SCI-style hybrid-engine has rollout+trainer sharing GPUs → requires reverting our disjoint-pool design. | Worth considering only after #1 if Qwen3 migration happens. |
| 27 | **SGLang instead of vLLM for rollout**. | Uncertain; ~similar throughput on most models | ROLLOUT | Very high (rewire tool-call path, test Qwen3.5 support). | Dr.SCI actually uses SGLang for the GenRM verifier, not rollout. vLLM is the verl default. No clear reason to switch. |
| 28 | **Rollout TP=4 → TP=1 + more DP nodes** (retool style with small model on standard transformer). | Rollout speedup, not trainer; not the bottleneck anyway. | ROLLOUT | Medium. | retool TP=1 is possible because Qwen2.5-7B fits in one A100's 40 GB KV cache. Qwen3.5-4B + 20k context: might or might not fit; need to test. |
| 29 | **Hybrid engine (rollout+trainer share GPUs)** instead of `hybrid_engine=False`. Frees dedicated rollout pool to be used as trainer pool. | Repacks same total GPUs — net effect depends on memory | TRAIN-TIME / ROLLOUT | High. | Dr.SCI / retool / dapo all do this. Trade-off: `rollout.gpu_memory_utilization` drops from 0.85 → 0.55 to leave room for trainer on the same GPUs. Would require rethinking the 5+1 het topology. |

### 12.6 Estimated compound effect (for planning)

Stacking only Tier-B + Tier-C levers (nothing Qwen3.5-blocked, nothing high-effort):

| Stack | Estimated step time |
|---|---|
| Baseline (step 1 actual) | 27:14 |
| + #14 + #15 + expandable_segments + 20k packed + (unblocks step 2 onwards) | ~24 min |
| + #6 (thinking 10,240) | ~21 min |
| + #10 (ref/rollout log-prob dynamic + ×4) | ~18 min |
| + #8 (staleness=1 overlap) | ~16 min |
| + #7 (2 trainer nodes, keep offload) | ~10–11 min |
| + #9 (disable offload, now safe at FSDP-8) | **~7–9 min** |

With Tier-A #1 (Qwen3 migration) on top: realistic floor ~4–5 min/step based on
retool/dapo's reported characteristics at similar model size.

Corresponding 48-hour step budgets:
- Baseline (if run didn't OOM): 27:14 → **~105 steps**.
- Tier-C only: ~21 min → ~135 steps.
- Tier-C + Tier-B conservative (#6+#10+#8): ~16 min → **~180 steps**.
- Tier-C + Tier-B + Tier-A #7 only: ~11 min → **~260 steps**.
- Full Tier-A #1+#2: ~5 min → **~570 steps**.

For reference, retool reaches AIME 0.60 in 150 GRPO steps. A 180–260 step budget
is above that reference-comparable threshold.

### 12.7 Caveats on these estimates

- No per-phase `timing_s/*` telemetry is currently emitted. The 22-min trainer
  decomposition is inferred from config (§4), not measured. Real numbers may differ.
- "Stacking" is approximate. Levers that hit the same bottleneck (e.g. activation
  memory) don't compound linearly. Two 20 % wins on activations ≠ 44 % win.
- Qwen3.5 SDPA cost estimate (~1.5–2× vs flash-attn) is from HF/benchmarks at
  similar sequence length; not measured on this cluster.
- FSDP-4 → FSDP-8 scaling depends on Perlmutter's Slingshot inter-node bandwidth;
  assumed ~linear based on typical reports, not verified.
- `filter_groups` compatibility with `fully_async_policy` is unverified (§9).
- Any change can surface a new bug on the experimental async path.

---

## 13. References

### Files (this repo)
- `CLAUDE.md` — Qwen3.5 overlay, attn_implementation=sdpa, wrap_policy notes.
- `scripts/perlmutter/smoke_tir_het.sbatch` — het topology + overrides submitted.
- `scripts/perlmutter/checklist.md` — smoke pass criteria and stop conditions.
- `scripts/perlmutter/_ray_bringup.sh` — Ray head + workers bring-up.
- `scripts/train_async.sh` — Hydra overrides for the async GRPO path.
- `verl/verl/experimental/fully_async_policy/fully_async_trainer.py` — async trainer.
- `perlmutter_debug/logs/physcode_smoke_tir_het_51693591.log` — main log.
- `perlmutter_debug/logs/physcode_smoke_tir_het_51693593.xverify.log` — xverify.
- `perlmutter_debug/outputs/.../rollout_dumps/1.jsonl` — step-1 dump (1024 rollouts).
- `outputs/probe_v5_A/rollouts.parquet`, `outputs/probe_v5_B/rollouts.parquet`
  — 10k probe rollouts each.
- `.claude/plans/physcode.md`, `.claude/plans/data-pipeline.md`,
  `.claude/plans/drsci-investigation.md` — project planning.
- `docs/training-decisions.md` — dataset/strategy notes referenced by CLAUDE.md.

### External repos (cloned)
- Dr.SCI — <https://github.com/MiniByte-666/Dr.SCI>
  at `/home/xs272/scratch/refs/Dr.SCI`.
  Launch script: `verl/examples/DR_SCI/final/qwen3_long_DrSCI_adaptive_difficulty_full.sh`.
- verl-recipe — <https://github.com/verl-project/verl-recipe>
  at `/home/xs272/scratch/refs/verl-recipe`.
  - retool: `retool/run_qwen2_7b_dapo.sh`, `retool/README.md`.
  - dapo:   `dapo/run_dapo_qwen2.5_32b.sh`, `dapo/README.md`.
  - r1:     `r1/run_r1_distill_qwen.sh`.

### External papers
- DAPO: Yu et al., "DAPO: An Open-Source LLM Reinforcement Learning System at
  Scale." arXiv:2503.14476.
- ReTool: "ReTool: Reinforcement Learning for Strategic Tool Use in LLMs."
  arXiv:2504.11536.
- Dr.SCI (original): Chen et al. "Improving Data and Reward Design for
  Scientific Reasoning in Large Language Models." arXiv:2602.08321.
- DeepSeek-R1 / R1-Zero — referenced for Clip-Higher baseline; arXiv:2501.12948.
- Meta ScaleRL / "The Art of Scaling RL for LLMs" — referenced for thinking-budget
  sizing precedent (~12–16 k for 8B+ models).

### Saved memory files referenced
- `memory/feedback_enforce_eager_degeneration.md` — Qwen3.5 `!!!` repetition under
  CUDA graph capture; use `enforce_eager=True`.
- `memory/feedback_no_priority_partitions.md` — Roberts priority partitions are
  billed at high rates; never submit there.
