"""Dedup + eval-contamination pipeline for Dr. SCI physics subset.

Loads data/processed/drsci_physics.parquet (115k rows), runs the same
3-pass dedup logic as run_dedup.py against the eval sets, and saves the
clean set to data/processed/drsci_physics_deduped.parquet.

Also reports overlap with the existing 6.8k training corpus
(candidates_deduped.parquet) — informational only, not removed.

Three passes:
  Pass 1 — Internal exact dedup of Dr. SCI rows (SHA-256 on question text)
  Pass 2 — Internal near-dedup via MinHash + LSH (Jaccard ≥ 0.8)
  Pass 3 — Contamination check against eval sets (exact + fuzzy)

Usage
-----
  python scripts/drsci_dedup.py
  python scripts/drsci_dedup.py --report          # stats only, no save
  python scripts/drsci_dedup.py --threshold 0.85  # stricter fuzzy threshold
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Re-use all dedup functions from run_dedup.py — no reimplementation.
from run_dedup import (  # noqa: E402 (sys.path must be set first)
    contamination_check,
    exact_dedup,
    fuzzy_dedup,
    load_eval_candidates,
    normalize_question,
    sha256_hash,
)
from phys_reasoner.data.schema import PhysicsProblem


# ---------------------------------------------------------------------------
# Load Dr. SCI parquet → list[PhysicsProblem]
# ---------------------------------------------------------------------------

def load_drsci(parquet_path: str) -> list[PhysicsProblem]:
    """Load drsci_physics.parquet and map to PhysicsProblem objects.

    Column mapping:
      problem_id  — synthetic "drsci_{index}"
      problem     — extra_info.question
      answer      — reward_model.ground_truth
      answer_type — "unknown" (inferred later in audit step)
      source      — "DrSCI"
      difficulty  — str(extra_info.difficulty)
      domain      — extra_info.subject (usually "physics")
      language    — "en" (filtered upstream)
      metadata    — extra_info.from, extra_info.difficulty (raw float)
    """
    df = pd.read_parquet(parquet_path)
    rows: list[PhysicsProblem] = []
    for idx, row in df.iterrows():
        question = str(row.get("extra_info.question") or row.get("question", ""))
        ground_truth = str(row.get("reward_model.ground_truth") or "")
        source_tag = str(row.get("extra_info.from") or "")
        difficulty_raw = row.get("extra_info.difficulty")
        difficulty_str = str(difficulty_raw) if difficulty_raw is not None else ""
        subject = str(row.get("extra_info.subject") or "physics")

        rows.append(
            PhysicsProblem(
                problem_id=f"drsci_{idx}",
                problem=question,
                answer=ground_truth,
                answer_type="unknown",
                source="DrSCI",  # type: ignore[arg-type]
                difficulty=difficulty_str,
                split="train_candidate",
                unit="",
                tolerance="",
                domain=subject,
                language="en",
                metadata={
                    "drsci_from": source_tag,
                    "drsci_difficulty_raw": float(difficulty_raw) if difficulty_raw is not None else None,
                    "drsci_original_index": int(idx),
                },
            )
        )
    return rows


# ---------------------------------------------------------------------------
# Save back to parquet — preserve original Dr. SCI columns + dedup metadata
# ---------------------------------------------------------------------------

def save_deduped(original_df: pd.DataFrame, kept_rows: list[PhysicsProblem], path: str) -> None:
    """Save deduped rows: original Dr. SCI columns for kept original indices."""
    kept_indices = [r.metadata["drsci_original_index"] for r in kept_rows]
    deduped_df = original_df.iloc[kept_indices].copy().reset_index(drop=True)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    deduped_df.to_parquet(path, index=False)
    print(f"Saved {len(deduped_df)} rows → {path}")


# ---------------------------------------------------------------------------
# Overlap report: Dr. SCI vs existing 6.8k training corpus
# ---------------------------------------------------------------------------

def report_overlap_with_corpus(
    drsci_rows: list[PhysicsProblem],
    corpus_path: str,
    threshold: float = 0.8,
) -> None:
    """Check how many Dr. SCI rows overlap with candidates_deduped.parquet.

    This is informational: overlapping rows are NOT removed from Dr. SCI here
    (they will be re-examined when building the final merged corpus).
    """
    if not Path(corpus_path).exists():
        print(f"  Corpus not found at {corpus_path} — skipping overlap check.")
        return

    print(f"  Loading existing corpus from {corpus_path}...", flush=True)
    corpus_df = pd.read_parquet(corpus_path)
    corpus_questions = [str(r) for r in corpus_df["problem"].tolist()]

    # Exact overlap
    corpus_hashes = {sha256_hash(normalize_question(q)) for q in corpus_questions}
    exact_overlap = sum(
        1 for r in drsci_rows
        if sha256_hash(normalize_question(r.problem)) in corpus_hashes
    )
    print(f"  Exact overlap with existing 6.8k corpus: {exact_overlap} / {len(drsci_rows)} "
          f"({exact_overlap / max(len(drsci_rows), 1):.1%})")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--input", default="data/processed/drsci_physics.parquet",
        help="Dr. SCI filtered parquet (default: data/processed/drsci_physics.parquet)",
    )
    parser.add_argument(
        "--output", default="data/processed/drsci_physics_deduped.parquet",
        help="Output path for deduplicated set",
    )
    parser.add_argument(
        "--corpus", default="data/processed/candidates_deduped.parquet",
        help="Existing training corpus — overlap is reported but rows are NOT removed",
    )
    parser.add_argument(
        "--cache_dir", default="data/hf_cache",
        help="HuggingFace cache dir (for loading eval sets)",
    )
    parser.add_argument(
        "--report", action="store_true",
        help="Print stats only — do not save output parquet",
    )
    parser.add_argument(
        "--threshold", type=float, default=0.8,
        help="Jaccard threshold for fuzzy dedup (default 0.8)",
    )
    args = parser.parse_args()

    print(f"Loading Dr. SCI rows from {args.input}...", flush=True)
    original_df = pd.read_parquet(args.input)
    rows = load_drsci(args.input)
    print(f"  Loaded {len(rows)} rows")

    # Source distribution
    source_counts = Counter(r.metadata.get("drsci_from", "") for r in rows)
    print("  Source distribution (extra_info.from):")
    for src, n in source_counts.most_common():
        print(f"    {src or '(blank)':<35} {n}")

    # Difficulty distribution
    diffs = [r.metadata.get("drsci_difficulty_raw") for r in rows
             if r.metadata.get("drsci_difficulty_raw") is not None]
    if diffs:
        import statistics
        print(f"  Difficulty — min={min(diffs):.3f}  median={statistics.median(diffs):.3f}  "
              f"max={max(diffs):.3f}  mean={statistics.mean(diffs):.3f}")

    # --- Pass 1: Internal exact dedup ---
    print("\nPass 1: Internal exact dedup...", flush=True)
    rows_after_exact, exact_pairs = exact_dedup(rows)
    print(f"  Removed: {len(exact_pairs)} exact duplicates")
    print(f"  Remaining: {len(rows_after_exact)} rows")

    pair_source_overlaps: Counter = Counter()
    for kept, removed in exact_pairs:
        kept_src = next((r.metadata.get("drsci_from", "?") for r in rows if r.problem_id == kept), "?")
        rem_src = next((r.metadata.get("drsci_from", "?") for r in rows if r.problem_id == removed), "?")
        pair_source_overlaps[(kept_src, rem_src)] += 1
    if pair_source_overlaps:
        print("  Per-source-pair exact overlap (within Dr. SCI):")
        for (s1, s2), n in pair_source_overlaps.most_common(10):
            print(f"    {s1} ↔ {s2}: {n}")

    # --- Pass 2: Internal near-dedup ---
    print(f"\nPass 2: Internal near-dedup (Jaccard ≥ {args.threshold})...", flush=True)
    rows_after_fuzzy, fuzzy_pairs = fuzzy_dedup(rows_after_exact, threshold=args.threshold)
    print(f"  Removed: {len(fuzzy_pairs)} near-duplicates")
    print(f"  Remaining: {len(rows_after_fuzzy)} rows")

    # --- Pass 3: Contamination check against eval sets ---
    print("\nPass 3: Eval contamination check...", flush=True)
    try:
        eval_rows = load_eval_candidates(args.cache_dir)
        print(f"  Eval rows loaded: {len(eval_rows)}")
        eval_source_counts = Counter(r.source for r in eval_rows)
        print("  Eval set breakdown:")
        for src, n in eval_source_counts.most_common():
            print(f"    {src:<30} {n}")

        rows_final, contam_pairs = contamination_check(
            rows_after_fuzzy, eval_rows, threshold=args.threshold,
        )
        pct = len(contam_pairs) / max(len(rows), 1)
        print(f"  Contaminated rows removed: {len(contam_pairs)} ({pct:.1%} of original)")
        print(f"  Clean rows remaining: {len(rows_final)}")

        if contam_pairs:
            # Which eval sets contributed most contamination
            eval_contam_src: Counter = Counter()
            eval_id_to_src = {r.problem_id: r.source for r in eval_rows}
            for _, eval_id, _ in contam_pairs:
                eval_contam_src[eval_id_to_src.get(eval_id, "?")] += 1
            print("  Contamination by eval source:")
            for src, n in eval_contam_src.most_common():
                print(f"    {src:<30} {n}")

        # Critical threshold check
        contam_pct = len(contam_pairs) / max(len(rows), 1)
        if contam_pct > 0.30:
            print(f"\n  WARNING: >30% contamination ({contam_pct:.1%}) — "
                  "dataset significantly smaller than expected.")
        else:
            print(f"\n  OK: contamination rate {contam_pct:.1%} is within acceptable range.")

    except Exception as e:
        print(f"  WARNING: contamination check failed: {e}")
        rows_final = rows_after_fuzzy

    # --- Overlap with existing corpus (informational) ---
    print("\nOverlap with existing training corpus (informational):")
    report_overlap_with_corpus(rows_final, args.corpus, threshold=args.threshold)

    # --- Summary ---
    print("\n" + "=" * 70)
    print("DEDUP SUMMARY")
    print("=" * 70)
    print(f"  Input rows:                   {len(rows)}")
    print(f"  After internal exact dedup:   {len(rows_after_exact)}  (-{len(exact_pairs)})")
    print(f"  After internal fuzzy dedup:   {len(rows_after_fuzzy)}  (-{len(fuzzy_pairs)})")
    print(f"  After eval contamination:     {len(rows_final)}")
    print(f"  Total removed:                {len(rows) - len(rows_final)}")
    print(f"  Retention rate:               {len(rows_final) / max(len(rows), 1):.1%}")
    print()

    # Post-dedup source distribution
    final_source_counts = Counter(r.metadata.get("drsci_from", "") for r in rows_final)
    print("  Final source distribution:")
    for src, n in final_source_counts.most_common():
        orig = source_counts.get(src, 0)
        print(f"    {src or '(blank)':<35} {n}  (was {orig})")

    # Decision signal
    n_final = len(rows_final)
    print()
    if n_final < 30_000:
        print(f"  CORPUS SIGNAL: {n_final:,} rows → Use as SUPPLEMENT to existing 6.8k")
    elif n_final < 80_000:
        print(f"  CORPUS SIGNAL: {n_final:,} rows → Dr. SCI becomes PRIMARY; 6.8k as hard-problem supplement")
    else:
        print(f"  CORPUS SIGNAL: {n_final:,} rows → Dr. SCI is PRIMARY; 6.8k moved to additional eval tier")
    print("=" * 70)

    if not args.report:
        save_deduped(original_df, rows_final, args.output)


if __name__ == "__main__":
    main()
