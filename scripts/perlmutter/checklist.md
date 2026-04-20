# Perlmutter Smoke Checklist & Stop Conditions

Short document — read top-to-bottom the first time, treat as a reference afterwards.

---

## Pre-submit (login node, one-time per session)

- [ ] `bash scripts/perlmutter/bootstrap.sh` returns `bootstrap PASSED`
- [ ] `data/processed_tir/data/{train,validation,test}.parquet` exist (from `fetch_dataset.py`)
- [ ] `WANDB_API_KEY` exported (or in `.env`), HF token already in `hf_cache` credentials
- [ ] `.env` has `SBATCH_ACCOUNT=<...>_g` if you want to avoid editing the sbatch file
- [ ] SLURM header in the sbatch matches the canonical Perlmutter values
      (`-A m2651`, `-C gpu&hbm80g`, `-q premium`). See `scripts/perlmutter/README.md` §2.

## Submit & observe (during smoke)

- [ ] `sbatch scripts/perlmutter/smoke_tir_het.sbatch` returns a job ID
- [ ] Within 3 min: `outputs/xverify_endpoints/current.url` contains a reachable URL
- [ ] From a login shell: `python3 scripts/perlmutter/probe_xverify.py` exits 0
- [ ] `ray status` (via `srun --overlap --het-group=0 -w <head> "${APT[@]}" ray status`)
      shows 6 nodes and 24 GPUs
- [ ] vLLM startup log (one per rollout node) shows `tensor_parallel_size=4`
      and you see **5 RolloutWorker actors** in Ray dashboard — confirms DP=5 × TP=4
      (not a single TP=20 blob)

## Smoke pass criteria (inspect after job finishes)

- [ ] `train.log` reaches `step=10` without errors and no python traceback
- [ ] `critic/score/mean > 0.0` by step 5 (reward signal is live)
- [ ] At least one rollout logged with both `[code]` and `\boxed{}` (full TIR cycle)
- [ ] At least one "time is up" / interrupt fired (if `MAX_TOOL_TURNS>=1` budgeted tightly)
- [ ] `timing_s/reward < 5s` consistently (xverify not bottleneck)
- [ ] No OOM in any rollout or trainer rank
- [ ] Checkpoint directory `outputs/physcode_tir_smoke/<EXPERIMENT>/global_step_5` exists
- [ ] First 3 training steps dumped to `<EXPERIMENT>/rollout_dumps/*.jsonl`;
      eyeball 20 random rollouts — rule-score and xverify-score should mostly agree;
      investigate any "rule=0 xverify=1" pattern before scaling up

## Resume test (item 4 in perlmutter-transfer-handoff.md)

- [ ] Submit smoke with `SAVE_FREQ=5 TOTAL_STEPS=20`
- [ ] When you see `global_step=6` in train.log, `scancel <job>`
- [ ] Resubmit with `--export=ALL,EXPERIMENT=<same>` — log shows
      `Resuming from step=5` and finishes at step 20 without a restart artifact in val metrics

## Profiling (collect once per new cluster)

- [ ] Run `scripts/parse_train_log.py <smoke>/train.log` (if present) and record:
  - `timing_s/gen` (rollout wall time per step)
  - `timing_s/update_actor` (trainer step time)
  - `timing_s/reward` (xverify latency)
- [ ] Multiply `gen` by `TOTAL_STEPS/smoke_steps` → production rollout ETA
- [ ] Multiply `update_actor` similarly → trainer ETA floor
- [ ] Production ETA = max(rollout, trainer) since we're on-policy (staleness=0)

---

## Stop conditions for the production run

`TOTAL_STEPS=3000` is the sbatch **ceiling** (~3 epochs on the 108k pool), not a target.
Stop resubmitting when any TWO of these trigger:

1. **Val plateau** — `val/score/mean` flat within ±0.5% for ≥3 consecutive eval points (≥300 steps)
2. **Zero-advantage fraction** — `actor/zero_advantage_group_fraction` > 0.70
   (most groups are all-correct or all-wrong; remaining signal is in the tail)
3. **Response length drop** — `actor/response_length/clip_ratio` near 0 AND reward ceiling-bound
   (model no longer using budget; capability ceiling for the current setup)

A run that plateaus at 1500 and gets scancelled is **a successful run** — it's not
abandoned compute. Don't chase the 3000 step count for its own sake.

---

## Finding a safe `PPO_MAX_TOKEN_LEN_PER_GPU` (micro-batch budget)

We use `actor.use_dynamic_bsz=True` + `ppo_max_token_len_per_gpu=N` instead of a fixed
micro-batch size. Principled sizing:

1. Smoke at `PPO_MAX_TOKEN_LEN_PER_GPU=24576` (≈ 1.25 × max_response_length).
2. If no OOM and `timing_s/update_actor` is stable, bump by ×1.5 → 36864, re-smoke.
3. Repeat until OOM, then back off 20%. That's your production value.
4. `log_prob_max_token_len_per_gpu` follows the same value (set both).

Typical A100-80G + FSDP-4 + optimizer-offload + grad-ckpt headroom at 20k ctx:
expect `~32k–48k` tokens/GPU per micro-batch. 40G is about half.

---

## xVerify sanity at runtime

- **Preflight**: `python3 scripts/perlmutter/probe_xverify.py` — all cases pass, p95<2s.
- **Loud failure** is already wired: `PHYS_REQUIRE_XVERIFY=1` in the sbatch makes
  `reward.py:_get_xverify_judge` raise `RuntimeError` if the URL isn't reachable,
  instead of silently falling back to rule-only.
- **Runtime canary**: first 3 training steps dumped to JSONL. Grep rollouts where
  `rule_score != xverify_score` — small disagreement is expected on expression types
  (that's why we use xverify), large disagreement on numericals indicates a bug.

---

## Common failures

| Symptom | Likely cause | Fix |
|---|---|---|
| `ray-probe` times out waiting for 6 nodes | Head IP unreachable from workers (firewall, wrong NIC) | Check `srun ... hostname --ip-address` output on head; try `NCCL_SOCKET_IFNAME=<nic>` |
| xverify URL file never appears | Het-group 2 stuck in queue, or `serve_xverify.sh` path wrong | Fall back to companion variant; check `logs/*.xverify.log` |
| OOM on step 1 but not step 0 | Peak activation is step-1 (cached optimizer state warms up) | Lower `PPO_MAX_TOKEN_LEN_PER_GPU` by 20%; or set `ACTOR_OPT_OFFLOAD=True` (should already be) |
| `gen` time grows each step | KV-cache pressure (vLLM preempting) | Raise `VLLM_GPU_MEM_UTIL` to 0.90; if already there, reduce `TRAIN_BATCH` or `THINKING_BUDGET` |
| Degenerate `!!!` rollouts | B200 TRTLLM bug | Should not apply on A100, but if seen, set `VLLM_USE_TRTLLM_ATTENTION=0` |
| Trainer GPU util near 0 during `gen` | Expected (on-policy; overlap requires `staleness_threshold>0`) | Not a bug |
