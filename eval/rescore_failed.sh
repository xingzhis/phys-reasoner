#!/usr/bin/env bash
# Find every <out_dir>/rollouts.parquet that lacks a scored.parquet next to it,
# and run the appropriate scorer (our_verifier for pool_v2, olympiad_official
# for OlympiadBench). Use after the master orchestrator finishes — earlier
# scoring runs may have crashed on individual rows (e.g. pint OffsetUnitCalculus
# on decibel) before the per-row try/except was added.
#
# Usage:
#   bash eval/rescore_failed.sh
#   CUDA_VISIBLE_DEVICES=0 bash eval/rescore_failed.sh   # pin xVerify GPU

set -uo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd -P)"

count=0
for D in $(find "$ROOT/outputs/eval" -mindepth 2 -maxdepth 2 -type d 2>/dev/null | sort); do
    R="$D/rollouts.parquet"
    S="$D/scored.parquet"
    [ -f "$R" ] || continue
    [ -f "$S" ] && continue
    count=$((count + 1))
    BENCH="$(basename "$(dirname "$D")")"
    case "$BENCH" in
        olympiad_*)  SCORER="olympiad_official" ;;
        pool_v2_*)   SCORER="our_verifier" ;;
        *) echo "skip unknown benchmark: $BENCH"; continue ;;
    esac
    LOG="$ROOT/logs/rescore_$(basename "$D")_${SCORER}.log"
    echo "[$count] $D  scorer=$SCORER  log=$LOG"

    # Match the orchestrator's score-stage env / binds.
    case "$SCORER" in
        our_verifier)
            ARGS=(--rollouts "$R" --out "$S" --xverify_model IAAR-Shanghai/xVerify-7B-I --xverify_device cuda)
            ;;
        olympiad_official)
            ARGS=(--rollouts "$R" --out "$S")
            ;;
    esac

    PYTHONNOUSERSITE=1 apptainer exec --nv \
      --overlay "$ROOT/phys-reasoner-overlay-017.img:ro" --no-home \
      --bind /etc/pki:/etc/pki --bind "$ROOT" \
      --bind "$(realpath "$ROOT/data")" \
      --bind "$(realpath "$ROOT/outputs")" \
      --bind "$(realpath "$ROOT/hf_cache")" \
      --pwd "$ROOT" \
      --env PYTHONNOUSERSITE=1 \
      --env "PYTHONPATH=$ROOT:$ROOT/src:$ROOT/eval/_pkgs:/opt/phys-extras/" \
      --env "HF_HOME=$ROOT/hf_cache" \
      --env "HF_DATASETS_OFFLINE=1" \
      --env "XDG_CACHE_HOME=/tmp/.cache" \
      --env "XDG_CONFIG_HOME=/tmp/.config" \
      ${CUDA_VISIBLE_DEVICES:+--env "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"} \
      "$ROOT/verl_vllm017.latest.sif" \
      python3 -m "eval.scoring.${SCORER}" "${ARGS[@]}" > "$LOG" 2>&1
    rc=$?
    if [ $rc -eq 0 ]; then
        pass=$(grep '^pass@1' "$D/scored.summary.txt" | head -1 || true)
        echo "    OK  $pass"
    else
        echo "    FAILED (exit $rc) — see $LOG"
    fi
done
echo "done — $count rescore attempts"
