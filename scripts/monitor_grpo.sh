#!/usr/bin/env bash
# monitor_grpo.sh — table view of per-step GRPO health metrics from a verl train.log.
#
# Usage:
#   bash scripts/monitor_grpo.sh                     # auto-find latest physcode_prod_*.log
#   bash scripts/monitor_grpo.sh logs/some.log       # specific log file
#   WATCH=1 bash scripts/monitor_grpo.sh             # refresh every 30s
#
# Columns:
#   step      — global_step
#   reward    — critic/score/mean
#   info_frac — critic/group/informative_frac (groups with within-group variance)
#   AR / AW   — all_right_frac / all_wrong_frac
#   adv_std   — critic/advantages/std (gradient signal magnitude)
#   ppo_kl    — actor/ppo_kl (effective policy movement)
#   gn        — actor/grad_norm
#   lr        — actor/lr (effective LR after warmup)
#   resp_len  — response_length/mean
#   step_s    — wall seconds per training step

set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd -P)"

LOG="${1:-}"
if [[ -z "$LOG" ]]; then
    LOG=$(ls -t "$ROOT"/logs/physcode_prod_*.log 2>/dev/null | grep -v xverify | head -1)
fi
[[ -z "$LOG" || ! -f "$LOG" ]] && { echo "no log found"; exit 1; }

extract() {
    local f="$1"
    python3 - "$f" <<'PY'
import re, sys
log = sys.argv[1]
keys = [
    ("training/global_step",            "step"),
    ("critic/score/mean",               "reward"),
    ("critic/group/informative_frac",   "info"),
    ("critic/group/all_right_frac",     "AR"),
    ("critic/group/all_wrong_frac",     "AW"),
    ("critic/advantages/std",           "adv_std"),
    ("actor/ppo_kl",                    "ppo_kl"),
    ("actor/grad_norm",                 "gn"),
    ("actor/lr",                        "lr"),
    ("response_length/mean",            "resp_len"),
    ("timing_s/step",                   "step_s"),
]
print(f"  log: {log}")
print("  " + "─" * 110)
hdr = ("step", "reward", "info", "AR", "AW", "adv_std", "ppo_kl", "gn", "lr", "resp_len", "step_s")
print(f"  {hdr[0]:>4}  {hdr[1]:>6}  {hdr[2]:>5}  {hdr[3]:>5}  {hdr[4]:>5}  {hdr[5]:>7}  "
      f"{hdr[6]:>8}  {hdr[7]:>6}  {hdr[8]:>9}  {hdr[9]:>8}  {hdr[10]:>6}")
print("  " + "─" * 110)
with open(log) as fh:
    for line in fh:
        # Match BOTH async (training/global_step) and sync DAPO (step:N - ...)
        if "training/global_step" not in line and not re.search(r"\bstep:[0-9]+ -", line):
            continue
        out = {}
        # Sync DAPO uses "step:N" prefix; capture as global_step
        m_sync = re.search(r"\bstep:([0-9]+) -", line)
        if m_sync:
            out["step"] = float(m_sync.group(1))
        for full, short in keys:
            # values may be wrapped in np.float64(X) — match optional wrapper
            m = re.search(rf"{re.escape(full)}:(?:np\.\w+\()?([\-0-9eE.+]+)\)?", line)
            if m:
                try: out[short] = float(m.group(1))
                except: pass
        if "step" not in out: continue
        s = out
        def fmt(v, d=3, e=False):
            if v is None: return "-"
            if e or (abs(v) > 0 and (abs(v) < 1e-3 or abs(v) >= 1e4)): return f"{v:.2e}"
            return f"{v:.{d}f}"
        print(f"  {int(s.get('step',0)):>4}  {fmt(s.get('reward'),3):>6}  "
              f"{fmt(s.get('info'),3):>5}  {fmt(s.get('AR'),3):>5}  {fmt(s.get('AW'),3):>5}  "
              f"{fmt(s.get('adv_std'),3):>7}  {fmt(s.get('ppo_kl'),0,True):>8}  "
              f"{fmt(s.get('gn'),3):>6}  {fmt(s.get('lr'),0,True):>9}  "
              f"{fmt(s.get('resp_len'),0):>8}  {fmt(s.get('step_s'),1):>6}")
PY
}

show() {
    clear 2>/dev/null || true
    echo "GRPO MONITOR — $(date '+%Y-%m-%d %H:%M:%S')"
    extract "$LOG"
    echo ""
    echo "  legend: info_frac >0.5 healthy, ppo_kl >1e-3 real movement, AR-AW > 0.3 saturating"
}

if [[ "${WATCH:-0}" == "1" ]]; then
    while true; do
        show
        sleep 30
    done
else
    show
fi
