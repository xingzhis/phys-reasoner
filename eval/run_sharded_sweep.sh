#!/usr/bin/env bash
# Generalized eval sweep: 5 benchmarks (4 pool_v2 sources + OlympiadBench OE_TO),
# each rollout sharded across both GPUs.
#
# Per benchmark we:
#   1. Pre-materialize the benchmark parquet (single-pass loader, no GPU needed).
#   2. Spawn 2 rollout shards in parallel:
#        GPU 0  → rows [0 : N/2)
#        GPU 1  → rows [N/2 : N)
#      Each shard writes its own rollouts.parquet under <base>/shard_{0,1}/.
#   3. Wait for both shards, concat shard rollouts.parquet → <base>/rollouts.parquet
#      (problem_idx is preserved as the original-pre-shard index by rollout.py
#      so the concat is correct without renumbering.)
#   4. Score the merged parquet on GPU 0 (xVerify-7B / sympy is the only GPU /
#      CPU consumer here, runs as subprocess so vLLM CUDA is freed first).
#
# Benchmarks run sequentially (smallest first) but each benchmark's heavy
# stage (rollout) uses both GPUs.
#
# Sampling preset (mode-agnostic — applies to TIR + CoT identically):
#   PRESET=train  → temperature=1.0 top_p=1.0 top_k=-1 repetition_penalty=1.0
#                   (matches training-time sampling; default)
#   PRESET=qwen   → temperature=0.6 top_p=0.95 top_k=20 repetition_penalty=1.0
#                   (Qwen team-recommended thinking-mode params)
# Or override individual {TEMPERATURE, TOP_P, TOP_K, REPETITION_PENALTY} envs.
#
# Budgets: response-length budget identity (matches VeRL training):
#   response_length = THINKING + INTERRUPT + TOOL_CALL + TOOL_RESPONSE + ANSWER
# Total response_length is identical between TIR and CoT modes (CoT collapses
# tool_call + tool_response + answer into a single post-interrupt budget).
#
# Usage:
#   MODEL=Qwen/Qwen3.5-4B MODE=tir PRESET=train TAG=train \
#     bash eval/run_sharded_sweep.sh
#   MODEL=Qwen/Qwen3.5-4B MODE=cot PRESET=qwen TAG=qwen \
#     bash eval/run_sharded_sweep.sh

set -uo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd -P)"
TAG="${TAG:-$(date +%Y%m%d.%H%M%S)}"

MODEL="${MODEL:-Qwen/Qwen3-4B-Thinking-2507}"
MODE="${MODE:-tir}"
N_ROLLOUTS="${N_ROLLOUTS:-1}"
GPU_MEM="${GPU_MEM:-0.7}"
THINKING_BUDGET="${THINKING_BUDGET:-12288}"
TOOL_CALL_BUDGET="${TOOL_CALL_BUDGET:-2048}"
ANSWER_BUDGET="${ANSWER_BUDGET:-4096}"
CHUNK_SIZE="${CHUNK_SIZE:-64}"

# Sampling preset → (temp, top_p, top_k, rep_pen). Caller may override each.
PRESET="${PRESET:-train}"
case "$PRESET" in
    train)
        : "${TEMPERATURE:=1.0}" "${TOP_P:=1.0}" "${TOP_K:=-1}" "${REPETITION_PENALTY:=1.0}"
        ;;
    qwen)
        : "${TEMPERATURE:=0.6}" "${TOP_P:=0.95}" "${TOP_K:=20}" "${REPETITION_PENALTY:=1.0}"
        ;;
    *)
        echo "PRESET must be 'train' or 'qwen' (got '$PRESET')" >&2
        exit 2
        ;;
esac
export TEMPERATURE TOP_P TOP_K REPETITION_PENALTY

# All benchmarks. Order: smallest first so any pipeline issues surface cheap.
# scibench(153) < physics(191) < ugphysics(217) < olympiad(236) < abench_phy_a(400)
# ≤ abench_phy_b(400) < drsci(503) < phybench(1000).
#
# Override via BENCHMARKS env (comma-separated list of registered benchmark names
# from run_eval.py). Used by run_external_sweeps.sh to target only the external
# benchmarks in a separate pass.
if [ -n "${BENCHMARKS:-}" ]; then
    IFS=',' read -ra BENCHMARKS <<< "$BENCHMARKS"
else
    BENCHMARKS=(
        pool_v2_scibench
        pool_v2_physics
        pool_v2_ugphysics
        olympiad_oe_to_physics
        pool_v2_drsci
    )
fi
LOGDIR="$ROOT/logs"
mkdir -p "$LOGDIR"

# printf %s avoids the trailing newline that `echo` adds, which `tr -c`
# would otherwise convert into a stray trailing underscore in the slug.
MODEL_SLUG="$(printf %s "$MODEL" | tr '/' '-' | tr -c '[:alnum:]._-' _)"

# Helper: read row count from the materialized parquet using a tiny Python invocation
# inside the apptainer container so the path resolves identically.
# Common apptainer flags. The worktree's data/, hf_cache/, outputs/ are symlinks
# whose targets live under the sibling repo path; bind the realpath of each
# symlink target so they resolve inside the container.
_apptainer_flags() {
    cat <<EOF
--overlay $ROOT/phys-reasoner-overlay-017.img:ro --no-home
--bind /etc/pki:/etc/pki --bind $ROOT
--bind $(realpath "$ROOT/data")
--bind $(realpath "$ROOT/outputs")
--bind $(realpath "$ROOT/hf_cache")
--pwd $ROOT
--env PYTHONNOUSERSITE=1 --env PYTHONPATH=/opt/phys-extras/
EOF
}

_count_rows() {
    local pq="$1"
    # shellcheck disable=SC2046
    PYTHONNOUSERSITE=1 apptainer exec $(_apptainer_flags) \
      "$ROOT/verl_vllm017.latest.sif" \
      python3 -c "import pandas as pd; print(len(pd.read_parquet('$pq')))"
}

_concat_parquets() {
    local out="$1"; shift
    local args=$(printf ",'%s'" "$@" | sed "s/^,//")
    # shellcheck disable=SC2046
    PYTHONNOUSERSITE=1 apptainer exec $(_apptainer_flags) \
      "$ROOT/verl_vllm017.latest.sif" \
      python3 -c "
import pandas as pd
dfs = [pd.read_parquet(p) for p in [$args]]
total = sum(len(d) for d in dfs)
out = pd.concat(dfs, ignore_index=True)
print(f'merged {len(dfs)} parquets ({[len(d) for d in dfs]}) → {len(out)} rows ({total} expected)')
out.to_parquet('$out', index=False)
"
}

echo "=== sharded sweep: model=$MODEL mode=$MODE preset=$PRESET tag=$TAG"
echo "    sampling: temperature=$TEMPERATURE top_p=$TOP_P top_k=$TOP_K rep_pen=$REPETITION_PENALTY"
echo "    budgets:  thinking=$THINKING_BUDGET tool_call=$TOOL_CALL_BUDGET answer=$ANSWER_BUDGET"
echo "    benchmarks: ${BENCHMARKS[*]}"
SUMMARY_PATHS=()

for BENCH in "${BENCHMARKS[@]}"; do
    BASE="outputs/eval/${BENCH}/${MODE}__${MODEL_SLUG}__${TAG}"
    BASE_ABS="$ROOT/$BASE"
    PARQUET="data/processed/eval/${BENCH}.parquet"
    PARQUET_ABS="$ROOT/$PARQUET"
    LOG_BASE="$LOGDIR/${BENCH}_${MODE}_${TAG}"
    mkdir -p "$BASE_ABS"

    echo ""
    echo "===== $BENCH → $BASE ====="

    # ---- Stage 1: materialize parquet (idempotent) -----------------------------
    if [ ! -f "$PARQUET_ABS" ]; then
        echo "[$BENCH] stage 1/4: materialize parquet"
        BENCHMARK="$BENCH" MODEL="$MODEL" MODE="$MODE" \
          OUT_DIR="$BASE/_loadonly" \
          SKIP_ROLLOUT=1 SKIP_SCORE=1 \
          bash "$ROOT/eval/run_eval.sh" > "${LOG_BASE}_load.log" 2>&1
        rc=$?
        rm -rf "$BASE_ABS/_loadonly"  # cleanup placeholder
        if [ $rc -ne 0 ]; then
            echo "[$BENCH] LOAD FAILED (exit $rc) — see ${LOG_BASE}_load.log; skipping source"
            continue
        fi
    else
        echo "[$BENCH] stage 1/4: parquet already exists ($PARQUET)"
    fi

    NROWS=$(_count_rows "$PARQUET_ABS")
    HALF=$((NROWS / 2))
    echo "[$BENCH] $NROWS rows → shard split: [0:$HALF) on GPU 0, [$HALF:$NROWS) on GPU 1"

    # ---- Stage 2: parallel shard rollouts -------------------------------------
    # N.B. cannot reuse the name `N` here — `run_eval.sh` reads `N` for sample
    # size (default -1 = all rows), and bash command-prefix env assignments
    # apply left-to-right within the same expansion, which would clobber the
    # row count we want for END_IDX. Use NROWS / HALF separately.
    echo "[$BENCH] stage 2/4: launch 2 shard rollouts in parallel"
    SHARD0_END="$HALF"; SHARD1_START="$HALF"; SHARD1_END="$NROWS"
    CUDA_VISIBLE_DEVICES=0 BENCHMARK="$BENCH" MODEL="$MODEL" MODE="$MODE" \
      N=-1 N_ROLLOUTS="$N_ROLLOUTS" GPU_MEM="$GPU_MEM" \
      THINKING_BUDGET="$THINKING_BUDGET" TOOL_CALL_BUDGET="$TOOL_CALL_BUDGET" \
      ANSWER_BUDGET="$ANSWER_BUDGET" CHUNK_SIZE="$CHUNK_SIZE" \
      START_IDX=0 END_IDX="$SHARD0_END" \
      OUT_DIR="$BASE/shard_0" SKIP_SCORE=1 \
      bash "$ROOT/eval/run_eval.sh" > "${LOG_BASE}_shard0.log" 2>&1 &
    PID0=$!

    CUDA_VISIBLE_DEVICES=1 BENCHMARK="$BENCH" MODEL="$MODEL" MODE="$MODE" \
      N=-1 N_ROLLOUTS="$N_ROLLOUTS" GPU_MEM="$GPU_MEM" \
      THINKING_BUDGET="$THINKING_BUDGET" TOOL_CALL_BUDGET="$TOOL_CALL_BUDGET" \
      ANSWER_BUDGET="$ANSWER_BUDGET" CHUNK_SIZE="$CHUNK_SIZE" \
      START_IDX="$SHARD1_START" END_IDX="$SHARD1_END" \
      OUT_DIR="$BASE/shard_1" SKIP_SCORE=1 \
      bash "$ROOT/eval/run_eval.sh" > "${LOG_BASE}_shard1.log" 2>&1 &
    PID1=$!

    wait $PID0; rc0=$?
    wait $PID1; rc1=$?
    if [ $rc0 -ne 0 ] || [ $rc1 -ne 0 ]; then
        echo "[$BENCH] SHARD ROLLOUT FAILED (rc0=$rc0 rc1=$rc1); skipping source"
        continue
    fi

    # ---- Stage 3: merge shard parquets ----------------------------------------
    echo "[$BENCH] stage 3/4: merge shard rollouts"
    if ! _concat_parquets "$BASE_ABS/rollouts.parquet" \
                          "$BASE_ABS/shard_0/rollouts.parquet" \
                          "$BASE_ABS/shard_1/rollouts.parquet"; then
        echo "[$BENCH] MERGE FAILED; skipping score"
        continue
    fi

    # ---- Stage 4: score merged rollouts on GPU 0 ------------------------------
    echo "[$BENCH] stage 4/4: score merged rollouts"
    CUDA_VISIBLE_DEVICES=0 BENCHMARK="$BENCH" MODEL="$MODEL" MODE="$MODE" \
      OUT_DIR="$BASE" SKIP_LOAD=1 SKIP_ROLLOUT=1 \
      bash "$ROOT/eval/run_eval.sh" > "${LOG_BASE}_score.log" 2>&1
    rc=$?
    if [ $rc -ne 0 ]; then
        echo "[$BENCH] SCORE FAILED (exit $rc) — see ${LOG_BASE}_score.log"
        continue
    fi

    SUMMARY="$BASE_ABS/scored.summary.txt"
    if [ -f "$SUMMARY" ]; then
        SUMMARY_PATHS+=("$SUMMARY")
        echo "[$BENCH] OK — summary at $SUMMARY"
    fi
done

echo ""
echo "=== pool_v2 sharded sequence done. Summaries: ==="
for s in "${SUMMARY_PATHS[@]}"; do
    echo "--- $s ---"
    cat "$s"
    echo ""
done
