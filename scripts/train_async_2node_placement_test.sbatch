#!/bin/bash
# train_async_2node_placement_test.sbatch — minimal 2-node × 1-GPU smoke that
# validates the trainer-coordinator placement fix (trainer_node Ray resource).
# Not a training run — 3 steps, 1 GPU per node. Submit to scavenge_gpu so we
# get fast turnaround; preemption risk is acceptable for a placement check.
#
# Expected signal: grep "[FullyAsyncTrainer] placed on" in the driver log —
# should print the WORKER hostname (= NODES[1], where trainer_node resource
# is advertised). If it prints the HEAD hostname, the fix is not working.
# Copy of train_async_2node.sbatch with shrunk resources.
#
# Original header below:
# train_async_2node.sbatch — 2-node async GRPO rehearsal for Perlmutter transfer.
#
# Topology:
#   Node 0 (HEAD  ): Ray head + 4x A100 for rollout (vLLM engine)
#   Node 1 (WORKER): Ray worker + 4x A100 for trainer (FSDP actor + ref)
#   Driver runs on HEAD; workers are scheduled by Ray onto the correct node via
#   placement groups (rollout→HEAD GPUs, trainer→WORKER GPUs).
#
# Why this exists:
#   scripts/train_smoke_async_misha.sbatch only exercised single-node 1+1 on a
#   2-GPU node — NCCL cross-node, Ray head/worker handshake, and param sync over
#   fabric are NOT exercised by that path. This sbatch is the multi-node rehearsal
#   before the Perlmutter bring-up (2026-04-13).
#
# Approach:
#   1. Request 2 nodes of 4x A100. qos_nmi and gpu partition come from .env via
#      SBATCH_PARTITION / SBATCH_QOS (exported by env.sh).
#   2. Resolve HEAD / WORKER hostnames from $SLURM_JOB_NODELIST.
#   3. Start `ray start --head` on HEAD inside apptainer.
#   4. Start `ray start --address=HEAD_IP:6379` on WORKER inside apptainer.
#   5. Poll Ray until cluster reports 2 alive nodes.
#   6. Run train_async.sh on HEAD with RAY_ADDRESS set → its python picks up the
#      existing cluster via ray.init() reading RAY_ADDRESS.
#   7. On exit, ray stop --force on both nodes.
#
# Submission:
#   sbatch scripts/train_async_2node.sbatch
#
# Overrides (env pass-through via --export=ALL):
#   TOTAL_STEPS, TRAIN_BATCH, ROLLOUT_N — test size
#   MODEL                                — default Qwen/Qwen3.5-4B
#   XVERIFY_URL                          — if set, trainer calls xVerify server
#   TIME_LIMIT                           — defaults to 3 hours
#
#SBATCH --job-name=async2node_placement
#SBATCH --partition=gpu_b200
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --gres=gpu:b200:1
#SBATCH --time=1:00:00
#SBATCH --output=logs/%x_%j.log
#SBATCH --error=logs/%x_%j.err
#
# Submit overrides for the resume rehearsal (gap #2 from perlmutter handoff):
#   sbatch --export=ALL,EXPERIMENT=resume_test_v1,SAVE_FREQ=2,TOTAL_STEPS=6 \
#          scripts/train_async_2node.sbatch
# Then scancel mid-run and resubmit with the same EXPERIMENT — verl's
# trainer.resume_mode=auto picks up the latest checkpoint in default_local_dir.

set -euo pipefail

ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
cd "$ROOT"
mkdir -p logs

echo "=== train_async_2node.sbatch ==="
echo "  Job      : $SLURM_JOB_ID"
echo "  Nodelist : $SLURM_JOB_NODELIST"
echo "  Time     : $(date -Iseconds)"

# Sourced env (HF_HOME, SIF, OVERLAY). Unset inherited SIF/OVERLAY in case of
# stale values from a parent shell (known pitfall per CLAUDE.md).
unset SIF OVERLAY
source "$ROOT/env.sh"

if [[ ! -f "$OVERLAY" ]]; then
    echo "ERROR: overlay not found: $OVERLAY" >&2
    exit 1
fi
if [[ ! -f "$SIF" ]]; then
    echo "ERROR: SIF not found: $SIF" >&2
    exit 1
fi
echo "  SIF      : $SIF"
echo "  OVERLAY  : $OVERLAY"
echo "  HF_HOME  : $HF_HOME"

# ---------------------------------------------------------------------------
# Spawn xVerify reward server NOW that we have a training allocation. This
# avoids the race where a separately-submitted xverify hits its wall-time
# before training ever leaves the queue. xverify writes its URL to
# outputs/xverify_endpoints/current.url which is polled below.
#
# Skip if XVERIFY_URL is already set (manual override) or XVERIFY_DISABLE=1.
# ---------------------------------------------------------------------------
XVERIFY_JOBID=""
if [[ -z "${XVERIFY_URL:-}" && "${XVERIFY_DISABLE:-0}" != "1" ]]; then
    # Clean stale URL file from previous runs.
    rm -f "$ROOT/outputs/xverify_endpoints/current.url"
    # Spawn the companion on a less-contended partition than the main training
    # job. .env's SBATCH_PARTITION is usually the heavy-GPU partition (gpu_h200,
    # gpu_b200) which is queue-stuck for whole-node jobs. xverify only needs 1
    # GPU and a 7B model fits on l40s/rtx_5000_ada/etc. Override at the spawn
    # site so the companion always lands somewhere it can actually start.
    XVERIFY_PARTITION="${XVERIFY_PARTITION:-scavenge_gpu}"
    XVERIFY_GRES="${XVERIFY_GRES:-gpu:l40s:1}"
    XVERIFY_JOBID=$(sbatch --parsable \
        --partition="$XVERIFY_PARTITION" \
        --gres="$XVERIFY_GRES" \
        "$ROOT/scripts/serve_xverify.sbatch" 2>&1 || echo "")
    if [[ "$XVERIFY_JOBID" =~ ^[0-9]+$ ]]; then
        echo "[xverify] submitted companion job $XVERIFY_JOBID on $XVERIFY_PARTITION ($XVERIFY_GRES)"
    else
        echo "[xverify] WARNING: sbatch failed: $XVERIFY_JOBID"
        XVERIFY_JOBID=""
    fi
fi

# Cleanup trap: cancel xverify on exit (success or failure) so we don't leak
# GPU quota. Also kill background srun ray steps if still alive.
cleanup() {
    local rc=$?
    echo "[cleanup] driver exit=$rc, tearing down..."
    if [[ -n "${XVERIFY_JOBID:-}" ]]; then
        echo "[cleanup] cancelling xverify job $XVERIFY_JOBID"
        scancel "$XVERIFY_JOBID" 2>/dev/null || true
    fi
    if [[ -n "${RAY_HEAD_PID:-}" ]]; then
        kill "$RAY_HEAD_PID" 2>/dev/null || true
    fi
    if [[ -n "${RAY_WORKER_PID:-}" ]]; then
        kill "$RAY_WORKER_PID" 2>/dev/null || true
    fi
    return $rc
}
trap cleanup EXIT

# ---------------------------------------------------------------------------
# Resolve node roles.
# ---------------------------------------------------------------------------
NODES=( $(scontrol show hostnames "$SLURM_JOB_NODELIST") )
if (( ${#NODES[@]} != 2 )); then
    echo "ERROR: expected 2 nodes, got ${#NODES[@]} (${NODES[*]})" >&2
    exit 1
fi
HEAD_NODE="${NODES[0]}"
WORKER_NODE="${NODES[1]}"

# Resolve HEAD IP from the HEAD node itself (its internal NIC address).
HEAD_IP=$(srun --nodes=1 --ntasks=1 -w "$HEAD_NODE" \
          hostname --ip-address | awk '{print $1}')

RAY_PORT=6379
DASHBOARD_PORT=8265
RAY_ADDRESS="${HEAD_IP}:${RAY_PORT}"
export RAY_ADDRESS

echo "  HEAD     : $HEAD_NODE ($HEAD_IP)"
echo "  WORKER   : $WORKER_NODE"
echo "  RAY      : $RAY_ADDRESS  (dashboard: $HEAD_IP:$DASHBOARD_PORT)"

# ---------------------------------------------------------------------------
# Apptainer prefix — same on every node.
# ---------------------------------------------------------------------------
APPTAINER_CMD=(
    apptainer exec --nv
    --overlay "$OVERLAY:ro"
    --no-home
    --bind /etc/pki:/etc/pki
    --env "PYTHONNOUSERSITE=1"
    --env "PYTHONUNBUFFERED=1"
    --env "PYTHONPATH=$ROOT/.async-extras:/opt/phys-extras/"
    --env "HF_HOME=$HF_HOME"
    --env "XDG_CACHE_HOME=/tmp/.cache"
    # flashinfer hardcodes ~/.cache/flashinfer at import time; with --no-home that
    # path is unwritable. FLASHINFER_WORKSPACE_BASE redirects it. HOME=/tmp is a
    # belt-and-suspenders default for any other library that checks $HOME.
    --env "FLASHINFER_WORKSPACE_BASE=/tmp"
    --env "HOME=/tmp"
    --env "TRITON_CACHE_DIR=/tmp/.cache/triton"
    --env "MPLCONFIGDIR=/tmp/.cache/matplotlib"
    # Production-mode xverify enforcement. When set to 1, any Ray actor whose
    # reward function fails to discover a live xverify server raises loudly
    # instead of silently falling back to rule-only. Must be forwarded to the
    # Ray daemon env so that actors spawned by the daemon inherit it.
    --env "PHYS_REQUIRE_XVERIFY=${PHYS_REQUIRE_XVERIFY:-0}"
    "$SIF"
)

# ---------------------------------------------------------------------------
# 1. Start Ray head on HEAD.
#    --block keeps the srun step alive; we background the entire srun.
# ---------------------------------------------------------------------------
echo "[ray] starting head on $HEAD_NODE ..."
srun --nodes=1 --ntasks=1 -w "$HEAD_NODE" --output="logs/%x_%j.ray_head.log" \
    "${APPTAINER_CMD[@]}" \
    ray start --head \
        --node-ip-address="$HEAD_IP" \
        --port=$RAY_PORT \
        --dashboard-host=0.0.0.0 \
        --dashboard-port=$DASHBOARD_PORT \
        --num-gpus="${N_GPUS_ROLLOUT:-1}" \
        --block &
RAY_HEAD_PID=$!

# Give head a moment to bind before workers attach.
sleep 15

# ---------------------------------------------------------------------------
# 2. Start Ray worker on WORKER.
# ---------------------------------------------------------------------------
echo "[ray] starting worker on $WORKER_NODE ..."
srun --nodes=1 --ntasks=1 -w "$WORKER_NODE" --output="logs/%x_%j.ray_worker.log" \
    "${APPTAINER_CMD[@]}" \
    ray start \
        --address="$RAY_ADDRESS" \
        --num-gpus="${N_GPUS_TRAIN:-1}" \
        --resources='{"trainer_node": 1}' \
        --block &
RAY_WORKER_PID=$!

# ---------------------------------------------------------------------------
# 3. Wait for cluster to have 2 alive nodes with 8 GPUs.
# ---------------------------------------------------------------------------
echo "[ray] waiting for 2-node cluster ..."
# --overlap lets this srun share the HEAD task slot with the backgrounded
# `ray start --head --block` srun above. Without it, slurm queues this step
# behind the ray-head srun and the cluster probe never runs (silent hang).
srun --overlap --nodes=1 --ntasks=1 -w "$HEAD_NODE" \
    "${APPTAINER_CMD[@]}" \
    python3 -c "
import ray, time, sys
ray.init(address='$RAY_ADDRESS')
for i in range(60):
    nodes = [n for n in ray.nodes() if n['Alive']]
    gpus  = sum(int(n['Resources'].get('GPU', 0)) for n in nodes)
    print(f'[ray-probe] attempt {i}: {len(nodes)} nodes, {gpus} gpus', flush=True)
    expected_gpus = ${N_GPUS_ROLLOUT:-1} + ${N_GPUS_TRAIN:-1}
    if len(nodes) >= 2 and gpus >= expected_gpus:
        print(f'[ray-probe] READY: {len(nodes)} nodes, {gpus} gpus', flush=True)
        sys.exit(0)
    time.sleep(2)
print('[ray-probe] FAILED: cluster did not reach 2 nodes / 8 gpus', flush=True)
sys.exit(1)
" || { echo "ERROR: ray cluster bootstrap failed" >&2; kill $RAY_HEAD_PID $RAY_WORKER_PID 2>/dev/null || true; exit 1; }

# ---------------------------------------------------------------------------
# 4. Launch the driver on HEAD.
#    train_async.sh calls apptainer internally; RAY_ADDRESS is forwarded via
#    the --env edit in train_async.sh so ray.init() connects to the existing
#    cluster instead of starting a new local one.
# ---------------------------------------------------------------------------
# Default smoke size: 3 steps, batch 4, rollout 4 → 12 rollouts total.
export TOTAL_STEPS="${TOTAL_STEPS:-3}"
export TRAIN_BATCH="${TRAIN_BATCH:-4}"
export ROLLOUT_N="${ROLLOUT_N:-4}"
export MODEL="${MODEL:-Qwen/Qwen3.5-4B}"
export N_GPUS_ROLLOUT="${N_GPUS_ROLLOUT:-1}"
export N_GPUS_TRAIN="${N_GPUS_TRAIN:-1}"
export NNODES_ROLLOUT=1
export NNODES_TRAIN=1
# Save/resume controls. Default -1 = no checkpointing (smoke runs).
# For the resume rehearsal pass SAVE_FREQ=2 and a stable EXPERIMENT name.
export SAVE_FREQ="${SAVE_FREQ:--1}"
export EXPERIMENT="${EXPERIMENT:-}"
# Rehearsal data: probe_subset (~2,500 stratified rows from train_split) is small
# enough that the dataloader doesn't dominate iter time, while still exercising
# every answer_type / source / difficulty bucket. Override TRAIN_FILES for production.
export TRAIN_FILES="${TRAIN_FILES:-$ROOT/data/processed/probe_subset.parquet}"
export VAL_FILES="${VAL_FILES:-$ROOT/data/processed/corpus_dev.parquet}"
export THINKING_BUDGET="${THINKING_BUDGET:-12288}"
export TOOL_CALL_BUDGET="${TOOL_CALL_BUDGET:-2048}"
export ANSWER_BUDGET="${ANSWER_BUDGET:-4096}"
export ACTOR_OPT_OFFLOAD=True
export VLLM_GPU_MEM_UTIL="${VLLM_GPU_MEM_UTIL:-0.85}"
export LOGGERS="${LOGGERS:-console}"

# ---------------------------------------------------------------------------
# Resolve XVERIFY_URL via rendezvous file written by serve_xverify.sbatch.
# If the caller already set XVERIFY_URL, honor that. Otherwise poll for up
# to XVERIFY_WAIT_SECS seconds (default 900 = 15 min) for the file to appear.
# If it never appears, fall back to rule-based reward only (empty URL).
# ---------------------------------------------------------------------------
if [[ -z "${XVERIFY_URL:-}" && "${XVERIFY_DISABLE:-0}" != "1" ]]; then
    URL_FILE="$ROOT/outputs/xverify_endpoints/current.url"
    # Bumped from 900 to 1200s to give companion xverify time to schedule + load.
    XVERIFY_WAIT_SECS="${XVERIFY_WAIT_SECS:-1200}"
    echo "[xverify] polling for $URL_FILE (timeout ${XVERIFY_WAIT_SECS}s) ..."
    deadline=$(( $(date +%s) + XVERIFY_WAIT_SECS ))
    while (( $(date +%s) < deadline )); do
        if [[ -f "$URL_FILE" ]]; then
            XVERIFY_URL=$(head -n1 "$URL_FILE")
            echo "[xverify] found URL: $XVERIFY_URL"
            break
        fi
        sleep 10
    done
    if [[ -n "${XVERIFY_URL:-}" ]]; then
        # Health check: wait up to 5 minutes for the server to start serving.
        echo "[xverify] health checking $XVERIFY_URL ..."
        health_url="${XVERIFY_URL%/judge}/health"
        hc_deadline=$(( $(date +%s) + 300 ))
        while (( $(date +%s) < hc_deadline )); do
            if curl -sf --max-time 5 "$health_url" >/dev/null 2>&1; then
                echo "[xverify] healthy"
                break
            fi
            sleep 10
        done
    else
        echo "[xverify] WARNING: URL file never appeared — continuing with rule-based reward only"
        XVERIFY_URL=""
    fi
fi
export XVERIFY_URL

echo "[driver] starting train_async.sh on $HEAD_NODE ..."
# --overlap: share HEAD task slot with the ray-head --block srun.
srun --overlap --nodes=1 --ntasks=1 -w "$HEAD_NODE" \
    bash "$ROOT/scripts/train_async.sh"
DRIVER_EXIT=$?

echo "[driver] exited with $DRIVER_EXIT"

# ---------------------------------------------------------------------------
# 5. Teardown.
# ---------------------------------------------------------------------------
echo "[ray] stopping cluster ..."
for N in "$HEAD_NODE" "$WORKER_NODE"; do
    srun --overlap --nodes=1 --ntasks=1 -w "$N" \
        "${APPTAINER_CMD[@]}" \
        ray stop --force || true
done

# Let the backgrounded ray --block srun steps exit after ray stop.
wait $RAY_HEAD_PID 2>/dev/null || true
wait $RAY_WORKER_PID 2>/dev/null || true

echo "=== done at $(date -Iseconds) (driver exit=$DRIVER_EXIT) ==="
exit $DRIVER_EXIT
