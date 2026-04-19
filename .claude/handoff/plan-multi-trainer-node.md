# Multi-trainer-node plan (Phase 4 — not active yet)

> **Scope:** this is for later. Phase-0 smoke must pass first. Do not implement until the user greenlights Phase 4.

## Why this might matter

`perlmutter_debug/FINDINGS.md` Tier-A lever #2: going from 1 trainer node × 4 GPU (FSDP=4) to 2 trainer nodes × 4 GPU (FSDP=8) lets us drop `optimizer_offload=True`, which was saving CPU↔GPU PCIe roundtrips per step. Dr.SCI runs FSDP-8 with no offload on the same 4B model size. Estimated combined speedup: 1.8–2.5× trainer phase.

New failure modes: cross-node NCCL on Perlmutter Slingshot. The Tier-A #3 (Ulysses SP) is already eating the cross-GPU comms budget on the single-node case; adding inter-node FSDP adds another comms layer.

## What the change touches

1. **`scripts/perlmutter/_ray_bringup.sh`** — currently starts Ray head on `HEAD_NODE` and workers on `WORKER_NODES` (rollout pool only). Need a second worker loop for a new `TRAIN_WORKER_NODES` array.
2. **`scripts/perlmutter/smoke_tir_het.sbatch` (or new `prod_tir_het_2trainer.sbatch`)** — het-group 0 goes from `-N 1` to `-N 2`.
3. **Hydra overrides in `train_async.sh`** — `trainer.nnodes=2` (env-var exposed already as `NNODES_TRAIN`); try `ACTOR_OPT_OFFLOAD=False` once FSDP-8 memory fits.
4. **NCCL tuning** — may need to revisit the env vars once we actually have cross-node FSDP traffic.

## Execution sketch (implement only after greenlight)

1. Write a scratch branch `feat/2-trainer-node` off `switch/qwen3-thinking`.
2. Extend `_ray_bringup.sh`: accept a `TRAIN_WORKER_NODES` array, loop over it with a second `srun` to start Ray workers on trainer nodes. Reuse the existing `--het-group` and placement-group logic.
3. Add a second sbatch variant or parameterize the existing one. Leave `smoke_tir_het.sbatch` as-is (single trainer node) for rollback.
4. Run a smoke: same TOTAL_STEPS=10, `NNODES_TRAIN=2`, `ACTOR_OPT_OFFLOAD=True` first (memory-safe), then retry with `False`.
5. Diagnosis points:
   - Ray placement: `ray status` should show 2 trainer nodes + 5 rollout nodes.
   - NCCL init: grep for "NCCL INFO" and any "connection timeout"; may need `NCCL_IB_TIMEOUT` bump.
   - FSDP sharding: `grep "Sharded parameters" launcher.log` — expect 8-way.
6. Compare step-1 wall against Phase-0 baseline. Expect trainer phase 1.8–2.5× faster if clean; if worse, ask user before continuing.

## Don't

- Don't try FSDP-across-nodes for a 4B model on Slingshot with untuned NCCL — cross-node all-reduce per attention layer will dominate. This plan avoids that by keeping TP=1 on trainer and only sharding params across nodes via FSDP.
- Don't combine with `staleness_threshold=1` AND `ROLLOUT_ENFORCE_EAGER=False` AND `ACTOR_OPT_OFFLOAD=False` all in the same first run. One structural change at a time.
- Don't skip the smoke step — jump straight to a full production run. A bad FSDP-8 setup wastes 48h.

## Rollback

- Delete `feat/2-trainer-node` branch.
- The single-trainer-node sbatch remains untouched.
- No `main`-branch changes at any point.
