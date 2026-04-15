#!/usr/bin/env bash
# _ray_bringup.sh — shared Ray head/worker bring-up for Perlmutter sbatch jobs.
# SOURCED, not executed. Callers must pre-export:
#   APT_CMD        array-name holding the apptainer prefix command
#   HEAD_NODE      hostname of Ray head (trainer)
#   WORKER_NODES   array of rollout-worker hostnames
#   N_GPUS_TRAIN   GPUs on HEAD to register with Ray (trainer pool)
#   N_GPUS_ROLLOUT GPUs on each WORKER to register with Ray (rollout pool)
#   HET_SRUN       (optional) space-separated extra srun flags to target a het group,
#                  e.g. "--het-group=0". Empty for non-het allocations.
# Exports on success:
#   RAY_ADDRESS    "<head-ip>:<port>" for the driver to pick up
#   RAY_HEAD_PID   backgrounded srun PID running ray head (caller must wait/kill on exit)
#   RAY_WORKER_PIDS  bash-array of backgrounded srun PIDs for workers
#
# The --block flag keeps each `ray start` srun alive for the job duration; we background
# each srun and let the driver run via a separate `--overlap` srun. On exit, callers
# should `ray stop --force` on every node and then kill the backgrounded srun PIDs.

_assert_set() {
    local name="$1"
    if [[ -z "${!name+x}" ]]; then
        echo "ERROR: _ray_bringup.sh needs $name set by caller" >&2
        exit 1
    fi
}
_assert_set HEAD_NODE
_assert_set WORKER_NODES
_assert_set N_GPUS_TRAIN
_assert_set N_GPUS_ROLLOUT

: "${RAY_PORT:=6379}"
: "${DASH_PORT:=8265}"
: "${HET_SRUN_HEAD:=}"     # e.g. "--het-group=0"
: "${HET_SRUN_WORKER:=}"   # e.g. "--het-group=1"

# Resolve HEAD IP from inside the head node (picks the internal NIC).
HEAD_IP=$(srun $HET_SRUN_HEAD --nodes=1 --ntasks=1 -w "$HEAD_NODE" \
          hostname --ip-address | awk '{print $1}')
export RAY_ADDRESS="${HEAD_IP}:${RAY_PORT}"

echo "[ray] HEAD=$HEAD_NODE ($HEAD_IP) WORKERS=${WORKER_NODES[*]}"
echo "[ray] RAY_ADDRESS=$RAY_ADDRESS  dashboard=$HEAD_IP:$DASH_PORT"

# -------- start head --------
srun $HET_SRUN_HEAD --nodes=1 --ntasks=1 -w "$HEAD_NODE" \
     --output="logs/%x_%j.ray_head.log" \
     "${APT_CMD[@]}" \
     ray start --head \
         --node-ip-address="$HEAD_IP" \
         --port="$RAY_PORT" \
         --dashboard-host=0.0.0.0 \
         --dashboard-port="$DASH_PORT" \
         --num-gpus="$N_GPUS_TRAIN" \
         --block &
RAY_HEAD_PID=$!
export RAY_HEAD_PID

sleep 15   # let head bind before workers attach

# -------- start workers --------
RAY_WORKER_PIDS=()
for W in "${WORKER_NODES[@]}"; do
    srun $HET_SRUN_WORKER --nodes=1 --ntasks=1 -w "$W" \
         --output="logs/%x_%j.ray_worker_${W}.log" \
         "${APT_CMD[@]}" \
         ray start \
             --address="$RAY_ADDRESS" \
             --num-gpus="$N_GPUS_ROLLOUT" \
             --block &
    RAY_WORKER_PIDS+=($!)
done
export RAY_WORKER_PIDS

# -------- wait for cluster to be fully up --------
EXPECTED_NODES=$((1 + ${#WORKER_NODES[@]}))
EXPECTED_GPUS=$((N_GPUS_TRAIN + ${#WORKER_NODES[@]} * N_GPUS_ROLLOUT))
echo "[ray] waiting for ${EXPECTED_NODES} nodes / ${EXPECTED_GPUS} GPUs ..."
srun --overlap $HET_SRUN_HEAD --nodes=1 --ntasks=1 -w "$HEAD_NODE" \
     "${APT_CMD[@]}" \
     python3 -c "
import ray, time, sys
ray.init(address='${RAY_ADDRESS}')
for i in range(120):
    nodes = [n for n in ray.nodes() if n['Alive']]
    gpus  = int(sum(n['Resources'].get('GPU', 0) for n in nodes))
    print(f'[ray-probe] attempt {i}: {len(nodes)} nodes, {gpus} gpus', flush=True)
    if len(nodes) >= ${EXPECTED_NODES} and gpus >= ${EXPECTED_GPUS}:
        print('[ray-probe] READY', flush=True); sys.exit(0)
    time.sleep(2)
sys.exit(1)
" || { echo "ERROR: ray cluster bootstrap failed" >&2; exit 1; }

# Handy teardown helper callers can invoke in a trap.
ray_teardown() {
    echo "[ray] stopping cluster ..."
    srun --overlap $HET_SRUN_HEAD --nodes=1 --ntasks=1 -w "$HEAD_NODE" \
         "${APT_CMD[@]}" ray stop --force 2>/dev/null || true
    for W in "${WORKER_NODES[@]}"; do
        srun --overlap $HET_SRUN_WORKER --nodes=1 --ntasks=1 -w "$W" \
             "${APT_CMD[@]}" ray stop --force 2>/dev/null || true
    done
    kill "$RAY_HEAD_PID" 2>/dev/null || true
    for p in "${RAY_WORKER_PIDS[@]}"; do kill "$p" 2>/dev/null || true; done
}
