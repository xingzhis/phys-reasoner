# Overnight Status — 2026-04-11

**Written:** 2026-04-11 ~01:30 local, before you sleep.

## TL;DR when you wake up

Run these to see where things ended up:
```bash
squeue -u $USER -o "%12i %15j %8T %10r %5D %12l %8M %R"
ls -la logs/train_async_2no*.log logs/xverify_server*.log
tail -60 logs/train_async_2no*.log
```

## What I submitted

| Job | ID | Script | Resources | Status at submit |
|---|---|---|---|---|
| xVerify server | **1493643** | `scripts/serve_xverify.sbatch` (new) | 1× l40s, 6h | RUNNING on r4513u10n01 |
| 2-node async training | **1493648** | `scripts/train_async_2node.sbatch` (new) | 2× h100:4, 6h | PENDING (Resources) |

## The plan change you need to know about

**qos_nmi has per-GPU-type caps I didn't know:**
- `a100: 4` (hard max — **cannot** do 2-node A100 via qos_nmi)
- `a40: 0` (blocked entirely for qos_nmi)
- `h100: 44`, `h200: 8`, `l40s: 12`
- Total across all types: 12 GPUs

**Implication:** the multi-node rehearsal cannot run on A100 under qos_nmi. I switched
to **2× H100** for correctness (distributed code path is GPU-class-agnostic). xVerify
uses L40S to keep H100 quota for training.

**A100 calibration deferred.** When you wake up, if multi-node passed, submit a
single-node A100 run separately for rollout calibration — `sbatch scripts/train_smoke_async_misha.sbatch`
WON'T work (stale node pins). You'll want to either:
- Copy `train_async_2node.sbatch` to a 1-node variant with `--nodes=1 --gres=gpu:a100:4`
  and change `N_GPUS_ROLLOUT=2 N_GPUS_TRAIN=2` (or 1+1 on a 4-GPU node)
- Or run interactive via `srun --pty --partition=gpu --qos=qos_nmi --gres=gpu:a100:4 ...`

## Files changed / created

1. `scripts/train_async.sh` — one-line edit: added `--env "RAY_ADDRESS=${RAY_ADDRESS:-}"` so
   `ray.init()` inside the apptainer picks up the external cluster address. This is the
   fix that makes multi-node work at all.

2. `scripts/train_async_2node.sbatch` — NEW. 2-node h100 async sbatch with Ray head/worker
   bootstrap and xVerify URL rendezvous. Bumped to 6h wall-time on your instruction.

3. `scripts/serve_xverify.sbatch` — NEW. Cluster-generic xVerify sbatch. Replaces stale
   `serve_xverify_misha.sbatch` which pinned to `--partition=test --nodelist=misha00`
   (partition doesn't exist). Writes its URL to
   `outputs/xverify_endpoints/current.url` for training to discover.

4. `docs/overnight-status-2026-04-11.md` — this file.

## Files NOT changed (still need your attention)

- `scripts/bootstrap_perlmutter.sh` — still **does not exist**. Write next, once the
  multi-node rehearsal is green. Draft it in the same SIF+overlay so it carries over.
- `scripts/train_smoke_async_misha.sbatch` — stale. Leave alone or delete.
- `scripts/serve_xverify_misha.sbatch` — stale. Leave alone or delete.
- Checkpoint-resume: not tested in this run. Do as run #2 once run #1 is green. Requires
  `SAVE_FREQ=1 TOTAL_STEPS=2` plus a manual kill mid-step and resubmit with resume=auto.

## How the 2-node training sbatch works (for debugging)

1. SLURM allocates 2 h100 nodes.
2. `scontrol show hostnames` picks HEAD + WORKER.
3. `srun` on HEAD runs `ray start --head` inside apptainer, backgrounded. Log:
   `logs/train_async_2node_<JID>.ray_head.log`.
4. `srun` on WORKER runs `ray start --address=HEAD_IP:6379` inside apptainer,
   backgrounded. Log: `logs/train_async_2node_<JID>.ray_worker.log`.
5. Sanity probe on HEAD waits for cluster to show 2 alive nodes + 8 GPUs, times out
   after ~2 min.
6. Polls `outputs/xverify_endpoints/current.url` (timeout 15 min) and health-checks
   `/health` endpoint.
7. Runs `train_async.sh` on HEAD with `RAY_ADDRESS=$HEAD_IP:6379` exported →
   forwarded into apptainer via the train_async.sh edit → `ray.init()` connects to
   external cluster instead of spawning local.
8. Driver runs TOTAL_STEPS=3 TRAIN_BATCH=4 ROLLOUT_N=4 on Qwen3.5-4B with full
   THINKING_BUDGET=12288 (the production budget that OOMed on single A40).
9. `ray stop --force` on both nodes on exit.

## What "success" looks like on run #1

Grep the main log for these markers:
- `[ray-probe] READY: 2 nodes, 8 gpus` — cluster bootstrap worked
- `[xverify] healthy` or `[xverify] WARNING` — rendezvous either worked or fell through
- `parameter sync` (or similar from verl async loop) — weights moved across the fabric
- `critic/score/mean` > 0 — reward signal fired
- Exit code 0 from the driver

## What to look at first if it fails

In priority order:
1. **If training never started** (stayed PENDING): check `squeue -u $USER` reason. If
   "Resources" → qos_nmi couldn't get 2 h100 nodes in 6h. Try `gpu:h200:4` instead
   (cap=8, same shape). Or split into 1+1 h100 with reduced batch.
2. **Ray bootstrap failed** (check `*.ray_head.log` / `*.ray_worker.log`): most common
   cause is worker couldn't reach HEAD_IP:6379. Could be firewall, could be Slingshot
   vs TCP routing on misha. Diagnostic: `srun -w WORKER_NODE ping HEAD_IP`.
3. **Ray OK, training OOM**: h100-80G should fit 4B comfortably with ACTOR_OPT_OFFLOAD.
   If it doesn't, enable `ACTOR_PARAM_OFFLOAD=True` via env override.
4. **Training runs but no parameter sync log**: that's the silent failure the handoff
   doc warned about. Means one-node-only despite `NNODES_*>1`. Double-check that
   `RAY_ADDRESS` actually landed inside apptainer (`env | grep RAY_ADDRESS` in log).

## Reproducing without waiting on me

```bash
# Check state
squeue -u $USER
cat outputs/xverify_endpoints/current.url     # current xverify URL
tail -60 logs/train_async_2node_*.log         # main driver log
tail -60 logs/train_async_2node_*.ray_head.log
tail -60 logs/train_async_2node_*.ray_worker.log

# Re-run training only (xverify already running)
sbatch scripts/train_async_2node.sbatch

# Submit both fresh
scancel -n xverify_server && sleep 5
sbatch scripts/serve_xverify.sbatch
sbatch scripts/train_async_2node.sbatch
```
