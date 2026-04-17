"""Rebuild the corpus training splits after dropping external benchmark sources.

Drops OlympiadBench, SciBench_RL, PHYBench, and PHYSICS from the corpus so they
can serve as clean external eval benchmarks. Only UGPhysics remains in training.
Dr. SCI splits are left untouched.

Pipeline (fail-loud at every step):
  1. Filter  data/processed/candidates_filtered.parquet → candidates_filtered_v2.parquet
     (keep UGPhysics only)
  2. Build   candidates_filtered_v2.parquet → corpus_train.parquet
     via scripts/build_training_parquets.py --corpus-only
  3. Split   corpus_train.parquet → corpus_{train_split,dev,test}.parquet
     via scripts/split_train_dev_test.py --corpus-only --corpus-stratum domain_coarse
  4. Verify  drsci_{train_split,dev,test}.parquet unchanged
  5. Sweep   all parquets for DROP_SOURCES contamination
  6. Merge   via scripts/merge_splits.py
  7. Verify  merged parquets clean

Usage:
    python3 scripts/rebuild_corpus_splits.py --report    # dry run (no writes)
    python3 scripts/rebuild_corpus_splits.py             # for real
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "processed"

KEEP_SOURCES = {"UGPhysics"}
DROP_SOURCES = {"PHYSICS", "OlympiadBench", "SciBench_RL", "PHYBench"}

# Expected row counts from current candidates_filtered.parquet (5414 UGPhysics rows).
# Allow a small ± tolerance in case upstream data is re-run.
EXPECTED_UGPHYSICS_ROWS = 5414
ROW_COUNT_TOLERANCE = 50


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _extract_source(df: pd.DataFrame) -> pd.Series:
    """Pull source tag out of either a flat 'source' col or extra_info struct."""
    if "source" in df.columns:
        return df["source"]
    if "extra_info" in df.columns:
        return df["extra_info"].apply(
            lambda x: x.get("source") if isinstance(x, dict) else None
        )
    return pd.Series([None] * len(df))


def _run(cmd: list[str], dry: bool) -> None:
    print(f"\n$ {' '.join(cmd)}")
    if dry:
        print("  (--report mode, skipped)")
        return
    result = subprocess.run(cmd, cwd=str(ROOT))
    if result.returncode != 0:
        raise SystemExit(f"FAIL: command returned {result.returncode}")


def _assert(cond: bool, msg: str) -> None:
    if not cond:
        raise SystemExit(f"ASSERTION FAILED: {msg}")
    print(f"  OK: {msg}")


# ---------------------------------------------------------------------------
# step 1: filter candidates
# ---------------------------------------------------------------------------

def step1_filter_candidates(report: bool) -> Path:
    print("\n" + "=" * 70)
    print("STEP 1: Filter candidates_filtered.parquet → candidates_filtered_v2.parquet")
    print("=" * 70)

    src = DATA_DIR / "candidates_filtered.parquet"
    dst = DATA_DIR / "candidates_filtered_v2.parquet"

    _assert(src.exists(), f"source file exists: {src}")

    df = pd.read_parquet(src)
    sources_before = df["source"].value_counts()
    print(f"  Input: {len(df):,} rows")
    print(f"  Source distribution:\n{sources_before.to_string()}")

    filtered = df[df["source"].isin(KEEP_SOURCES)].copy()
    n_drop = len(df) - len(filtered)
    print(f"\n  After filter: {len(filtered):,} rows (dropped {n_drop:,})")

    sources_after = set(filtered["source"].unique())
    _assert(sources_after == KEEP_SOURCES,
            f"filtered source set == {KEEP_SOURCES} (got {sources_after})")
    _assert(sources_after.isdisjoint(DROP_SOURCES),
            f"no DROP_SOURCES in filtered data (DROP={DROP_SOURCES})")
    _assert(abs(len(filtered) - EXPECTED_UGPHYSICS_ROWS) <= ROW_COUNT_TOLERANCE,
            f"row count {len(filtered)} within ±{ROW_COUNT_TOLERANCE} of "
            f"expected {EXPECTED_UGPHYSICS_ROWS}")

    # v2 is cheap (just filtered rows) and auditable. Always write it so
    # downstream steps can read it even in --report mode.
    filtered.to_parquet(dst, index=False)
    print(f"\n  Wrote: {dst}")

    return dst


# ---------------------------------------------------------------------------
# step 2: rebuild corpus_train.parquet
# ---------------------------------------------------------------------------

def step2_build_corpus(v2_path: Path, report: bool) -> Path:
    print("\n" + "=" * 70)
    print("STEP 2: Build corpus_train.parquet via build_training_parquets.py")
    print("=" * 70)

    out = DATA_DIR / "corpus_train.parquet"
    cmd = [
        sys.executable, str(ROOT / "scripts" / "build_training_parquets.py"),
        "--corpus-only",
        "--corpus-input", str(v2_path),
        "--corpus-output", str(out),
    ]
    if report:
        cmd.append("--report")
    _run(cmd, dry=False)  # the script itself respects --report

    if not report:
        _assert(out.exists(), f"corpus_train.parquet written: {out}")
        df = pd.read_parquet(out)
        sources = set(_extract_source(df).dropna().unique())
        _assert(sources == KEEP_SOURCES,
                f"corpus_train extra_info.source == {KEEP_SOURCES} (got {sources})")
        print(f"  corpus_train: {len(df):,} rows, sources={sources}")

    return out


# ---------------------------------------------------------------------------
# step 3: split
# ---------------------------------------------------------------------------

def step3_split(report: bool) -> None:
    print("\n" + "=" * 70)
    print("STEP 3: Split corpus via split_train_dev_test.py --corpus-stratum domain_coarse")
    print("=" * 70)

    cmd = [
        sys.executable, str(ROOT / "scripts" / "split_train_dev_test.py"),
        "--corpus-only",
        "--corpus-stratum", "domain_coarse",
        "--corpus-dev", "200",
        "--corpus-test", "200",
        "--seed", "42",
    ]
    if report:
        cmd.append("--report")
    _run(cmd, dry=False)

    if not report:
        ts = pd.read_parquet(DATA_DIR / "corpus_train_split.parquet")
        dv = pd.read_parquet(DATA_DIR / "corpus_dev.parquet")
        te = pd.read_parquet(DATA_DIR / "corpus_test.parquet")
        tot = pd.read_parquet(DATA_DIR / "corpus_train.parquet")
        _assert(len(ts) + len(dv) + len(te) == len(tot),
                f"train_split + dev + test = corpus_train ({len(ts)}+{len(dv)}+{len(te)}={len(tot)})")
        _assert(len(dv) == 200, f"corpus_dev has exactly 200 rows (got {len(dv)})")
        _assert(len(te) == 200, f"corpus_test has exactly 200 rows (got {len(te)})")


# ---------------------------------------------------------------------------
# step 4: verify Dr. SCI unchanged
# ---------------------------------------------------------------------------

def step4_verify_drsci(report: bool) -> None:
    print("\n" + "=" * 70)
    print("STEP 4: Verify Dr. SCI splits unchanged")
    print("=" * 70)

    expected = {
        "drsci_train_split.parquet": (95_000, 102_000),  # ~98.5k
        "drsci_dev.parquet":         (2_000, 2_000),
        "drsci_test.parquet":        (2_000, 2_000),
    }
    for name, (lo, hi) in expected.items():
        p = DATA_DIR / name
        _assert(p.exists(), f"{name} exists (must NOT regenerate these)")
        n = len(pd.read_parquet(p))
        _assert(lo <= n <= hi, f"{name} size {n} in [{lo}, {hi}]")


# ---------------------------------------------------------------------------
# step 5: contamination sweep
# ---------------------------------------------------------------------------

def step5_contamination_sweep(report: bool) -> None:
    print("\n" + "=" * 70)
    print("STEP 5: Contamination sweep — DROP_SOURCES must not appear anywhere")
    print("=" * 70)

    files = [
        "corpus_train.parquet",
        "corpus_train_split.parquet",
        "corpus_dev.parquet",
        "corpus_test.parquet",
        "drsci_train.parquet",
        "drsci_train_split.parquet",
        "drsci_dev.parquet",
        "drsci_test.parquet",
    ]
    any_leak = False
    for name in files:
        p = DATA_DIR / name
        if not p.exists():
            print(f"  SKIP (missing): {name}")
            continue
        df = pd.read_parquet(p)
        sources = set(_extract_source(df).dropna().unique())
        leaked = sources & DROP_SOURCES
        if leaked:
            any_leak = True
            print(f"  CONTAMINATION in {name}: {leaked}")
        else:
            print(f"  OK: {name} ({len(df):,} rows, sources={sources})")

    _assert(not any_leak, "no DROP_SOURCES leaked into ANY split")
    print("\n  CONTAMINATION CHECK PASSED")


# ---------------------------------------------------------------------------
# step 6: merge
# ---------------------------------------------------------------------------

def step6_merge(report: bool) -> None:
    print("\n" + "=" * 70)
    print("STEP 6: Merge splits via merge_splits.py")
    print("=" * 70)

    cmd = [sys.executable, str(ROOT / "scripts" / "merge_splits.py")]
    _run(cmd, dry=report)

    if report:
        return

    out_dir = DATA_DIR / "merged"
    for split in ["train", "validation", "test"]:
        p = out_dir / f"{split}.parquet"
        _assert(p.exists(), f"merged {split}.parquet written")
        df = pd.read_parquet(p)
        sources = set(_extract_source(df).dropna().unique())
        leaked = sources & DROP_SOURCES
        _assert(not leaked, f"merged/{split}.parquet clean (sources={sources})")
        curated = df[df["pool"] == "curated"]
        print(f"  merged/{split}: {len(df):,} total, curated={len(curated):,}")


# ---------------------------------------------------------------------------
# step 7: probe_subset untouched
# ---------------------------------------------------------------------------

def step7_probe_untouched(report: bool, start_mtime: float | None) -> None:
    print("\n" + "=" * 70)
    print("STEP 7: Verify probe_subset.parquet not modified")
    print("=" * 70)

    p = DATA_DIR / "probe_subset.parquet"
    if not p.exists():
        print(f"  SKIP: {p} does not exist")
        return
    mtime = p.stat().st_mtime
    if start_mtime is None:
        print(f"  probe_subset.parquet mtime={mtime} (no baseline recorded)")
        return
    _assert(mtime == start_mtime,
            f"probe_subset.parquet mtime unchanged (was {start_mtime}, now {mtime})")


# ---------------------------------------------------------------------------
# summary
# ---------------------------------------------------------------------------

def print_summary() -> None:
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    files = [
        ("corpus_train.parquet",       "corpus (rebuilt)"),
        ("corpus_train_split.parquet", "  → train_split"),
        ("corpus_dev.parquet",         "  → dev"),
        ("corpus_test.parquet",        "  → test"),
        ("drsci_train.parquet",        "drsci (unchanged)"),
        ("drsci_train_split.parquet",  "  → train_split"),
        ("drsci_dev.parquet",          "  → dev"),
        ("drsci_test.parquet",         "  → test"),
        ("merged/train.parquet",       "merged train"),
        ("merged/validation.parquet",  "merged validation"),
        ("merged/test.parquet",        "merged test"),
    ]
    for name, label in files:
        p = DATA_DIR / name
        if p.exists():
            n = len(pd.read_parquet(p))
            print(f"  {label:<30} {n:>10,} rows  ({p})")
        else:
            print(f"  {label:<30} MISSING")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--report", action="store_true",
                    help="Dry run: print what would happen, don't write files")
    args = ap.parse_args()

    probe = DATA_DIR / "probe_subset.parquet"
    start_probe_mtime = probe.stat().st_mtime if probe.exists() else None

    v2 = step1_filter_candidates(args.report)
    step2_build_corpus(v2, args.report)
    step3_split(args.report)
    if not args.report:
        step4_verify_drsci(args.report)
        step5_contamination_sweep(args.report)
    step6_merge(args.report)
    if not args.report:
        step7_probe_untouched(args.report, start_probe_mtime)
        print_summary()

    print("\nDone." + (" (--report mode: no files written)" if args.report else ""))


if __name__ == "__main__":
    main()
