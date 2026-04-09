# Profiling and Bottleneck Diagnosis — VeRL GRPO TIR Runs

**Purpose:** systematically identify where wall-clock time goes in a VeRL training step, and pick the right intervention based on the dominant phase.
**Companion to:** `.claude/plans/perlmutter-transfer.md`

---

## TL;DR — the 4-step protocol

1. **Decompose**: get per-phase time from `timing_s/*` in train.log. Skip warmup steps (0–4).
2. **Identify the dominant phase** (>40% of step time).
3. **Validate the bottleneck hypothesis**: check the resource that the dominant phase *should* be hitting (GPU util, KV cache, NCCL traffic, etc.). If the resource is not pegged, it's a different bottleneck than you think.
4. **Make ONE change**, re-measure. Don't change two things at once.

---

## Where time is spent in a VeRL step

VeRL logs `timing_s/<phase>` for every step. The phases (for the GRPO TIR loop):

| Phase | What it does | Resource it should hit |
|---|---|---|
| `gen` | vLLM rollout — generate responses for all prompts in the batch | Rollout GPU(s), KV cache memory |
| `reward` | Custom reward function — calls xVerify, parses `\boxed{}` | xVerify GPU + CPU |
| `ref` | Reference policy log probs over the response | Trainer GPU(s), forward only |
| `old_log_prob` | Current policy log probs over the response (no_grad) | Trainer GPU(s), forward only |
| `values` | (PPO only — N/A for GRPO) | — |
| `adv` | Compute GRPO advantages (CPU + small reductions) | CPU, NCCL all-reduce |
| `update_actor` | Backward + optimizer step | Trainer GPU(s), full forward+backward+opt |
| `update_critic` | (PPO only — N/A for GRPO) | — |
| (sync only on async) `param_sync` | Push trainer weights to rollouter pool | Cross-node bandwidth |

Total = sum of these. In **colocate sync** mode, all phases are serial. In **async on-policy** mode, `gen` runs on rollout GPUs while everything else runs on trainer GPUs — but on-policy with `staleness=0` means trainer waits for rollout before starting, so the effective wall-clock is still `gen + (ref + old_log_prob + adv + update_actor) + reward`. Async only buys true overlap with `staleness>0` (which we are NOT using initially).

---

## Reading the timing data

VeRL writes lines like:
```
step:5 timing_s/gen:42.13 timing_s/ref:0.51 timing_s/old_log_prob:2.07 timing_s/update_actor:8.21 timing_s/reward:0.39 timing_s/adv:0.04 ...
```

The `scripts/parse_train_log.py` helper (TODO #11 in plan) reads these, drops warmup steps, and prints:

```
Phase           mean    p50     p95     %tot
gen             42.1s   41.8s   45.3s   78%
ref              0.5s    0.5s    0.6s    1%
old_log_prob     2.1s    2.0s    2.3s    4%
update_actor     8.2s    8.0s    9.1s   15%
reward           0.4s    0.4s    0.5s    1%
total           53.8s   53.1s   57.8s  100%
```

**Read this immediately after every job.** It's the single most important diagnostic and it costs nothing.

### Sanity checks on the timing table

- `total` should match wall-time / num_steps within ~5%. If not, you're missing a phase (probably async waits).
- `gen` should be the dominant phase for long-context TIR (60–80% is normal).
- `update_actor` should be 10–25% of total. If it's >40%, the trainer is starving the GPU (small batches, optimizer offload, or gradient sync issues).
- `reward` should be <5% of total when xVerify is healthy. If >10%, the verifier is the bottleneck.
- `ref + old_log_prob` together should be <10%. If higher, log-prob micro batch is too small.

---

## Resource validation: is the dominant phase actually pegged?

Knowing `gen` takes 80% of step time isn't enough. You need to confirm `gen` is **actually compute/memory bound**, not waiting on something. The following table tells you what to look for and what to do.

### `gen` (rollout) is dominant

Open a side terminal and run during a step:
```bash
# Inside the apptainer environment, on a rollout node
nvidia-smi dmon -s pucvmet -d 2 -i 0,1,2,3
```

| Symptom | Diagnosis | Fix |
|---|---|---|
| GPU util ~100%, memory ~90% of `gpu_memory_utilization` setting | True compute-bound rollout. KV cache is being used effectively. | More rollout GPUs (scale up TP within node, then DP across nodes). Or shrink workload (smaller `THINKING_BUDGET`, smaller batch). |
| GPU util high, memory low (<50%) | KV cache is underused. Either `gpu_memory_utilization` is set too low, or sequence count too small to fill it. | Raise `gpu_memory_utilization` to 0.9. If already there, increase `max_num_seqs`. |
| GPU util ~50% bouncing, memory ~90% | KV cache pressure: vLLM is preempting and recomputing sequences. Look for `Preempted: N` in vLLM logs. | More KV cache headroom: more rollout GPUs (TP=4 instead of TP=2, or DP=2 instead of DP=1). Or shorter `THINKING_BUDGET`. |
| GPU util low across all rollout GPUs | Rollout is waiting on something — most likely the verifier (reward call) or a Ray sync. | Check `timing_s/reward`; check `ray status`; check that all DP rollout engines are alive. |
| Only one rollout GPU is busy in a TP=N setup | TP all-reduce is failing or one rank is stuck. | Restart the rollout pool. Check NCCL env vars and NVLink topology with `nvidia-smi topo -m`. |
| Rollout time has a long-tail (`p95/p50 > 1.5`) | A few sequences are running to the response budget (truncated). Expected with thinking traces. | Check `response_length/clip_ratio` in train.log; reduce `THINKING_BUDGET` if >0.5. |

### `update_actor` (trainer) is dominant

| Symptom | Diagnosis | Fix |
|---|---|---|
| GPU util high, NVLink traffic high, time scales linearly with batch | Compute-bound trainer. Normal. | Increase `PPO_MICRO_BS_PER_GPU` if memory allows; or scale FSDP across more GPUs (within a node). |
| GPU util low, time long | NCCL all-reduce stalling, or optimizer offload swapping CPU↔GPU. | Disable `ACTOR_OPT_OFFLOAD` if memory allows. Check NCCL with `NCCL_DEBUG=INFO`. |
| Memory error during update | Activation memory too high | Increase `enable_gradient_checkpointing=True` (already on); reduce `PPO_MICRO_BS_PER_GPU`. |
| `update_actor/p95` >> `p50` | Some steps trigger an extra synchronization (e.g., parameter sync to rollouter). | Time-align with `param_sync` phase. Likely benign. |

### `reward` (verifier) is dominant

| Symptom | Diagnosis | Fix |
|---|---|---|
| `timing_s/reward > 10s` | xVerify GPU is overloaded or rule-verifier is hitting xVerify too often | Profile xVerify call rate; verify the rule pre-check is firing for easy cases (numerical match). Add caching for repeated answers. |
| `reward` time grows with batch size | Sequential xVerify calls per rollout | xVerify client should batch, OR the trainer should call rewards in parallel via threadpool |
| `reward` time spikes randomly | Verifier server crash or network hiccup | Verifier client retry must be in place. Check xVerify server logs. |

### Cross-node async (Config C only): `param_sync` or generic waiting

| Symptom | Diagnosis | Fix |
|---|---|---|
| `param_sync` >5s per occurrence | Cross-node weight transfer is slow | Confirm Slingshot is being used; check `nvidia-smi nvlink` and rail counters; not much to do beyond reducing sync frequency (`trigger_parameter_sync_step=N>1`) |
| Trainer GPU util drops to 0 between batches, total step time grows | Trainer is waiting on rollout to finish; rollout is the bottleneck. The async overlap isn't happening because we're on-policy. | Either accept it (on-policy is correct for stable training), or increase `staleness_threshold` to 1 |
| One rollout DP engine never returns | Cross-node Ray placement group split-brain | `ray status`; restart job. Consider falling back to Config A for next run. |

---

## System-level observability commands

These run inside the apptainer environment. Add them to a `scripts/profile_node.sh` helper.

```bash
# GPU utilization sampled every 2s, all GPUs on this node
nvidia-smi dmon -s pucvmet -d 2

# Per-GPU memory snapshot
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv -l 5

# NVLink throughput (within-node TP comm)
nvidia-smi nvlink -gt d

# Topology — how GPUs and NICs are connected
nvidia-smi topo -m

# Process tree on this node
ps -ef --forest | grep -E "(python|vllm|ray)"

# Ray cluster health (from any node in the placement group)
ray status

# Slingshot per-NIC counters (need to verify on Perlmutter; ask collaborator)
# Probably something like: cxi_stat or rdma_perftest
```

---

## A worked example: how to use this on Apr 14

After the 50-step replication run, run `scripts/parse_train_log.py outputs/.../train.log` and you might see:

```
Phase           mean    p50     p95     %tot
gen             88.0s   85.4s  103.7s   85%
update_actor     9.5s    9.3s   10.2s    9%
ref              1.2s    1.1s   1.4s     1%
old_log_prob     3.8s    3.7s   4.1s     4%
reward           0.6s    0.5s   0.7s     1%
total          103.1s  100.0s 119.1s   100%
```

**Diagnosis**: `gen` is 85% of step time, much higher than the ~60s target. Open `nvidia-smi dmon` on the rollout node next run and check:
- If GPU util is ~100% and memory is ~90% → genuinely compute-bound, need more GPUs (we're already at TP=4 within a node) → switch from Config A to Config C (DP=2 across nodes).
- If GPU util is bouncing ~50% with high memory → KV cache pressure, vLLM preempting. Confirm via vLLM `Preempted: N` log line. Same fix: more rollout GPUs.
- If GPU util is low → something is blocking. Check for verifier hangs or Ray issues.

In all three cases for this example, the answer is "more rollout GPUs", which means **commit to Config C** for production.

---

## Anti-patterns

- ❌ Changing two knobs at once after seeing high `gen`. You won't know which one fixed it.
- ❌ Optimizing memory pressure by lowering `THINKING_BUDGET` without first checking if KV cache is actually pressured. You may be sacrificing reasoning quality unnecessarily.
- ❌ Adding more rollout nodes when GPU util on existing rollout nodes is already low. The bottleneck is elsewhere (verifier, sync, network).
- ❌ Running PyTorch profiler on a 4B model with 16k context. Heavy, slow, mostly tells you what `nvidia-smi` already told you. Reserve for surgical debugging of specific kernels.
- ❌ Trusting test-cluster numbers as-is on Perlmutter. NUMA, NICs, BIOS, ECC scrub all differ. Always re-measure on the new cluster.
- ❌ Skipping the warmup steps. Step 0 includes vLLM JIT compile, FSDP all-gather warmup, and HF cache loads. Always drop steps 0–4 from the timing table.

---

## Quick reference: when to use which Config

| Step time on Config A (single-node colocate) | Action |
|---|---|
| ≤ 70s | Stay on Config A. Multi-node not worth the complexity for a 1.5–2 day saving. |
| 70–100s | Switch to Config C (1 trainer + 2 rollout). Roughly halves rollout time. |
| > 100s | Switch to Config C AND profile per-phase to find the real issue. Possibly verifier or KV cache pressure. |

The Apr 13 bracket test gives you the actual numbers to make this decision.
