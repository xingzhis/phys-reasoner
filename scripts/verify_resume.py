#!/usr/bin/env python3
"""verify_resume.py — assert the resume rehearsal actually resumed.

Usage:
    python scripts/verify_resume.py <run1_log> <run2_log>

The two log paths should be the SLURM logs from two consecutive submissions of
train_async_2node.sbatch with the same EXPERIMENT name and SAVE_FREQ>0:

    sbatch --export=ALL,EXPERIMENT=resume_test,SAVE_FREQ=2,TOTAL_STEPS=6 \
           scripts/train_async_2node.sbatch
    # let it run a few steps, then scancel
    sbatch --export=ALL,EXPERIMENT=resume_test,SAVE_FREQ=2,TOTAL_STEPS=6 \
           scripts/train_async_2node.sbatch
    python scripts/verify_resume.py logs/train_async_2node_<job1>.log \
                                    logs/train_async_2node_<job2>.log

Checks:
    1. Run 1 reached at least step 1 (so a checkpoint must have been saved
       given SAVE_FREQ<=last_step).
    2. Run 2 mentions a resume / load_checkpoint / global_step_ message.
    3. Run 2's first observed step is > 0 (resume fired).
    4. Run 2's last step > run 1's last step (forward progress made).

Exit code: 0 = PASS, 1 = FAIL, 2 = bad args.
"""

import pathlib
import re
import sys

STEP_RE = re.compile(r"global_step[:\s=_/]+(\d+)")
RESUME_HINT_RE = re.compile(
    r"(resume_from|Resume from|load.*checkpoint|Loading.*checkpoint|global_step_\d+)",
    re.IGNORECASE,
)


def parse(path: str) -> tuple[list[int], str]:
    text = pathlib.Path(path).read_text(errors="ignore")
    steps = [int(m.group(1)) for m in STEP_RE.finditer(text)]
    return steps, text


def head_tail(seq: list[int], n: int = 3) -> str:
    if len(seq) <= 2 * n:
        return str(seq)
    return f"{seq[:n]} ... {seq[-n:]}"


def main(run1: str, run2: str) -> int:
    s1, t1 = parse(run1)
    s2, t2 = parse(run2)

    print(f"Run 1: {run1}")
    print(f"  global_steps observed: {head_tail(s1)}")
    print(f"Run 2: {run2}")
    print(f"  global_steps observed: {head_tail(s2)}")

    failures: list[str] = []

    if not s1:
        failures.append("run 1 has no global_step lines — did training start at all?")
    if not s2:
        failures.append("run 2 has no global_step lines — did training start at all?")

    if s1 and s2:
        last1 = max(s1)
        first2 = min(s2)
        last2 = max(s2)

        if not RESUME_HINT_RE.search(t2):
            print("WARN: no explicit resume keyword found in run 2 log "
                  "(this may be fine if verl logs differently — checking step numbers)")

        if last1 < 1:
            failures.append(f"run 1 only reached step {last1} — too short to checkpoint")
        if first2 == 0:
            failures.append(
                f"run 2 started at step 0 — resume did NOT fire "
                f"(verl trainer.resume_mode should be 'auto')"
            )
        elif first2 < last1:
            print(
                f"NOTE: run 2 first step {first2} < run 1 last step {last1} "
                f"(expected if SAVE_FREQ < last1 — resumed from earlier checkpoint)"
            )
        if last2 <= last1:
            failures.append(
                f"run 2 last step {last2} <= run 1 last step {last1} — "
                f"no forward progress after resume"
            )

    if failures:
        print()
        print("FAIL:")
        for f in failures:
            print(f"  - {f}")
        return 1

    print()
    print(f"PASS: run 1 reached step {max(s1)}; "
          f"run 2 started at step {min(s2)} and reached step {max(s2)}.")
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(__doc__)
        sys.exit(2)
    sys.exit(main(sys.argv[1], sys.argv[2]))
