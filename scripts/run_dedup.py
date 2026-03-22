"""All-pairs dedup pipeline for training candidates.

Three passes:
  Pass 1 — Exact dedup via SHA-256 on normalized question text (O(N), instant)
  Pass 2 — Near-dedup via MinHash + LSH at Jaccard threshold 0.8 (O(N) with datasketch or numpy fallback)
  Pass 3 — Train/eval contamination check (exact + fuzzy overlap with each eval set)

Priority order for keeping duplicates:
  OlympiadBench > PHYSICS > UGPhysics > PHYBench > SciBench_RL

Outputs:
  data/processed/candidates_deduped.parquet

Usage
-----
  python scripts/run_dedup.py
  python scripts/run_dedup.py --report   # print stats, don't save
  python scripts/run_dedup.py --threshold 0.85  # stricter fuzzy threshold
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from phys_reasoner.data.schema import PhysicsProblem
from phys_reasoner.data.normalize import normalize_answer_type

# ---------------------------------------------------------------------------
# Source priority (higher = keep when deduping)
# ---------------------------------------------------------------------------

_SOURCE_PRIORITY = {
    "OlympiadBench": 5,
    "PHYSICS":       4,
    "UGPhysics":     3,
    "PHYBench":      2,
    "SciBench_RL":   1,
}


# ---------------------------------------------------------------------------
# Load all training candidates from parquet
# ---------------------------------------------------------------------------

def load_candidates(parquet_path: str) -> list[PhysicsProblem]:
    import pandas as pd

    df = pd.read_parquet(parquet_path)
    rows = []
    for _, r in df.iterrows():
        answer = r["answer"]
        if isinstance(answer, str):
            try:
                parsed = json.loads(answer)
                if isinstance(parsed, list):
                    answer = parsed
            except (json.JSONDecodeError, TypeError):
                pass
        answer_type = r["answer_type"]
        if isinstance(answer_type, str):
            try:
                parsed = json.loads(answer_type)
                if isinstance(parsed, list):
                    answer_type = parsed
            except (json.JSONDecodeError, TypeError):
                pass
        rows.append(
            PhysicsProblem(
                problem_id=str(r["problem_id"]),
                problem=str(r["problem"]),
                answer=answer,
                answer_type=normalize_answer_type(answer_type, str(r.get("source", ""))),
                source=r["source"],
                difficulty=str(r.get("difficulty") or ""),
                split=str(r.get("split") or "train_candidate"),
                unit=str(r.get("unit") or ""),
                tolerance=str(r.get("tolerance") or ""),
                domain=str(r.get("domain") or ""),
                language=str(r.get("language") or "en"),
            )
        )
    return rows


def load_eval_candidates(cache_dir: str) -> list[PhysicsProblem]:
    """Load eval sets for contamination check (Pass 3)."""
    from phys_reasoner.data.loaders import load_critpt, load_yale_physics, load_abench

    rows: list[PhysicsProblem] = []
    print("Loading eval sets for contamination check...", flush=True)
    rows.extend(load_critpt(cache_dir=cache_dir))
    rows.extend(load_yale_physics(tier="textonly"))
    rows.extend(load_abench(subset="Phy_A"))
    rows.extend(load_abench(subset="Phy_B"))
    return rows


# ---------------------------------------------------------------------------
# Question normalization
# ---------------------------------------------------------------------------

_LATEX_WS_RE = re.compile(r"\s+")
_PUNCT_RE = re.compile(r"[^\w\s\\{}^_]")


def normalize_question(text: str) -> str:
    """Normalize question text for dedup comparison."""
    text = text.lower()
    # Normalize whitespace in LaTeX (e.g., \frac {a}{b} → \frac{a}{b})
    text = _LATEX_WS_RE.sub(" ", text)
    # Strip punctuation except LaTeX structural chars
    text = _PUNCT_RE.sub("", text)
    return text.strip()


def sha256_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Pass 1 — Exact dedup
# ---------------------------------------------------------------------------

def exact_dedup(
    rows: list[PhysicsProblem],
) -> tuple[list[PhysicsProblem], list[tuple[str, str]]]:
    """Remove exact duplicates (by normalized question hash).

    Returns (deduplicated_rows, duplicate_pairs) where each pair is
    (kept_problem_id, removed_problem_id).
    """
    hash_to_row: dict[str, PhysicsProblem] = {}
    duplicates: list[tuple[str, str]] = []
    kept: list[PhysicsProblem] = []

    for row in rows:
        h = sha256_hash(normalize_question(row.problem))
        if h in hash_to_row:
            existing = hash_to_row[h]
            # Keep higher-priority source
            if _SOURCE_PRIORITY.get(row.source, 0) > _SOURCE_PRIORITY.get(existing.source, 0):
                duplicates.append((row.problem_id, existing.problem_id))
                hash_to_row[h] = row
            else:
                duplicates.append((existing.problem_id, row.problem_id))
        else:
            hash_to_row[h] = row

    kept = list(hash_to_row.values())
    return kept, duplicates


# ---------------------------------------------------------------------------
# Pass 2 — Near-dedup via MinHash + LSH
# ---------------------------------------------------------------------------

def _char_ngrams(text: str, n: int = 5) -> set[str]:
    return {text[i : i + n] for i in range(len(text) - n + 1)}


def _minhash(shingles: set[str], num_perm: int = 128, seed: int = 42) -> np.ndarray:
    """Compute MinHash signature using numpy (no datasketch needed)."""
    rng = np.random.default_rng(seed)
    a = rng.integers(1, (1 << 31) - 1, size=num_perm, dtype=np.int64)
    b = rng.integers(0, (1 << 31) - 1, size=num_perm, dtype=np.int64)
    _MERSENNE = (1 << 31) - 1

    sig = np.full(num_perm, np.iinfo(np.int64).max, dtype=np.int64)
    for s in shingles:
        hv = int(hashlib.md5(s.encode()).hexdigest(), 16) % _MERSENNE
        candidate = (a * hv + b) % _MERSENNE
        sig = np.minimum(sig, candidate)
    return sig


def _lsh_bands(sig: np.ndarray, num_bands: int = 16) -> list[str]:
    """Split signature into bands and return band keys."""
    rows_per_band = len(sig) // num_bands
    keys = []
    for b in range(num_bands):
        start = b * rows_per_band
        end = start + rows_per_band
        band_bytes = sig[start:end].tobytes()
        keys.append(f"{b}:{hashlib.md5(band_bytes).hexdigest()}")
    return keys


def fuzzy_dedup(
    rows: list[PhysicsProblem],
    threshold: float = 0.8,
    num_perm: int = 128,
    num_bands: int = 16,
) -> tuple[list[PhysicsProblem], list[tuple[str, str, float]]]:
    """Remove near-duplicates via MinHash LSH.

    Returns (deduplicated_rows, fuzzy_pairs) where each pair is
    (kept_problem_id, removed_problem_id, estimated_jaccard).
    """
    print(f"  Computing MinHash signatures ({num_perm} perms, {num_bands} bands)...", flush=True)

    # Compute signatures
    sigs: list[np.ndarray] = []
    for row in rows:
        norm = normalize_question(row.problem)
        shingles = _char_ngrams(norm, n=5)
        sigs.append(_minhash(shingles, num_perm=num_perm))

    # LSH bucketing
    print("  Building LSH index...", flush=True)
    buckets: dict[str, list[int]] = defaultdict(list)
    for idx, sig in enumerate(sigs):
        for key in _lsh_bands(sig, num_bands=num_bands):
            buckets[key].append(idx)

    # Find candidate pairs
    print("  Finding candidate pairs...", flush=True)
    candidate_pairs: set[tuple[int, int]] = set()
    for bucket_members in buckets.values():
        if len(bucket_members) < 2:
            continue
        for i in range(len(bucket_members)):
            for j in range(i + 1, len(bucket_members)):
                a, b = bucket_members[i], bucket_members[j]
                if a > b:
                    a, b = b, a
                candidate_pairs.add((a, b))

    print(f"  Candidate pairs to verify: {len(candidate_pairs)}", flush=True)

    # Verify Jaccard similarity for candidate pairs
    to_remove: set[int] = set()
    fuzzy_pairs: list[tuple[str, str, float]] = []

    for a, b in candidate_pairs:
        if a in to_remove or b in to_remove:
            continue
        # Exact Jaccard on shingles
        norm_a = normalize_question(rows[a].problem)
        norm_b = normalize_question(rows[b].problem)
        shingles_a = _char_ngrams(norm_a, n=5)
        shingles_b = _char_ngrams(norm_b, n=5)
        union = len(shingles_a | shingles_b)
        if union == 0:
            continue
        jaccard = len(shingles_a & shingles_b) / union
        if jaccard >= threshold:
            # Keep higher-priority source
            row_a = rows[a]
            row_b = rows[b]
            pri_a = _SOURCE_PRIORITY.get(row_a.source, 0)
            pri_b = _SOURCE_PRIORITY.get(row_b.source, 0)
            if pri_a >= pri_b:
                to_remove.add(b)
                fuzzy_pairs.append((row_a.problem_id, row_b.problem_id, jaccard))
            else:
                to_remove.add(a)
                fuzzy_pairs.append((row_b.problem_id, row_a.problem_id, jaccard))

    kept = [r for i, r in enumerate(rows) if i not in to_remove]
    return kept, fuzzy_pairs


# ---------------------------------------------------------------------------
# Pass 3 — Train/eval contamination check
# ---------------------------------------------------------------------------

def contamination_check(
    train_rows: list[PhysicsProblem],
    eval_rows: list[PhysicsProblem],
    threshold: float = 0.8,
    num_perm: int = 128,
    num_bands: int = 16,
) -> tuple[list[PhysicsProblem], list[tuple[str, str, float]]]:
    """Remove training rows that appear in any eval set (exact + fuzzy).

    Returns (clean_train_rows, contaminated_pairs).
    """
    print("  Pass 3: checking train/eval contamination...", flush=True)

    # Exact contamination check
    eval_hashes = {sha256_hash(normalize_question(r.problem)) for r in eval_rows}
    exact_contaminated: set[str] = set()
    for r in train_rows:
        if sha256_hash(normalize_question(r.problem)) in eval_hashes:
            exact_contaminated.add(r.problem_id)

    print(f"    Exact contamination: {len(exact_contaminated)} rows", flush=True)

    # Fuzzy contamination check via MinHash
    all_rows = train_rows + eval_rows
    n_train = len(train_rows)

    sigs: list[np.ndarray] = []
    for row in all_rows:
        norm = normalize_question(row.problem)
        shingles = _char_ngrams(norm, n=5)
        sigs.append(_minhash(shingles, num_perm=num_perm))

    buckets: dict[str, list[int]] = defaultdict(list)
    for idx, sig in enumerate(sigs):
        for key in _lsh_bands(sig, num_bands=num_bands):
            buckets[key].append(idx)

    fuzzy_contaminated: set[str] = set()
    contaminated_pairs: list[tuple[str, str, float]] = []

    for bucket_members in buckets.values():
        train_in_bucket = [i for i in bucket_members if i < n_train]
        eval_in_bucket  = [i for i in bucket_members if i >= n_train]
        if not train_in_bucket or not eval_in_bucket:
            continue
        for ti in train_in_bucket:
            for ei in eval_in_bucket:
                norm_t = normalize_question(train_rows[ti].problem)
                norm_e = normalize_question(eval_rows[ei - n_train].problem)
                shingles_t = _char_ngrams(norm_t, n=5)
                shingles_e = _char_ngrams(norm_e, n=5)
                union = len(shingles_t | shingles_e)
                if union == 0:
                    continue
                jaccard = len(shingles_t & shingles_e) / union
                if jaccard >= threshold:
                    fuzzy_contaminated.add(train_rows[ti].problem_id)
                    contaminated_pairs.append(
                        (train_rows[ti].problem_id, eval_rows[ei - n_train].problem_id, jaccard)
                    )

    print(f"    Fuzzy contamination: {len(fuzzy_contaminated)} rows", flush=True)

    all_contaminated = exact_contaminated | fuzzy_contaminated
    clean = [r for r in train_rows if r.problem_id not in all_contaminated]
    all_pairs = (
        [(pid, "eval_exact_match", 1.0) for pid in exact_contaminated]
        + contaminated_pairs
    )
    return clean, all_pairs


# ---------------------------------------------------------------------------
# Save parquet
# ---------------------------------------------------------------------------

def save_parquet(rows: list[PhysicsProblem], path: str) -> None:
    import json
    import pandas as pd

    records = []
    for r in rows:
        d = r.to_dict()
        if isinstance(d.get("answer"), list):
            d["answer"] = json.dumps(d["answer"])
        if isinstance(d.get("answer_type"), list):
            d["answer_type"] = json.dumps(d["answer_type"])
        records.append(d)

    df = pd.DataFrame(records)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    print(f"Saved {len(df)} rows to {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="All-pairs dedup pipeline")
    parser.add_argument(
        "--input", default="data/processed/candidates_raw.parquet",
        help="Input parquet from explore_quality.py",
    )
    parser.add_argument(
        "--output", default="data/processed/candidates_deduped.parquet",
        help="Output parquet after dedup",
    )
    parser.add_argument(
        "--report", action="store_true",
        help="Print stats only, don't save output",
    )
    parser.add_argument(
        "--threshold", type=float, default=0.8,
        help="Jaccard threshold for fuzzy dedup (default 0.8)",
    )
    parser.add_argument(
        "--cache_dir", default="data/hf_cache",
        help="HuggingFace cache dir (for loading eval sets)",
    )
    args = parser.parse_args()

    print(f"Loading candidates from {args.input}...", flush=True)
    rows = load_candidates(args.input)
    print(f"  Loaded {len(rows)} training candidates")

    # Count by source
    from collections import Counter
    source_counts = Counter(r.source for r in rows)
    for src, n in sorted(source_counts.items()):
        print(f"    {src:<25} {n}")

    # --- Pass 1: Exact dedup ---
    print("\nPass 1: Exact dedup...", flush=True)
    rows_after_exact, exact_pairs = exact_dedup(rows)
    print(f"  Removed: {len(exact_pairs)} exact duplicates")
    print(f"  Remaining: {len(rows_after_exact)} rows")

    if exact_pairs:
        print("  Examples (kept → removed):")
        for kept, removed in exact_pairs[:5]:
            print(f"    {kept}  →  {removed}")

    # Per-pair source overlap report
    pair_overlaps: Counter = Counter()
    kept_ids = {r.problem_id for r in rows_after_exact}
    id_to_source = {r.problem_id: r.source for r in rows}
    for kept, removed in exact_pairs:
        src_kept = id_to_source.get(kept, "?")
        src_removed = id_to_source.get(removed, "?")
        pair_overlaps[(src_kept, src_removed)] += 1
    if pair_overlaps:
        print("  Per-source-pair exact overlap:")
        for (s1, s2), n in pair_overlaps.most_common():
            print(f"    {s1} ↔ {s2}: {n}")

    # --- Pass 2: Fuzzy dedup ---
    print(f"\nPass 2: Fuzzy dedup (Jaccard ≥ {args.threshold})...", flush=True)
    rows_after_fuzzy, fuzzy_pairs = fuzzy_dedup(
        rows_after_exact, threshold=args.threshold
    )
    print(f"  Removed: {len(fuzzy_pairs)} near-duplicates")
    print(f"  Remaining: {len(rows_after_fuzzy)} rows")

    if fuzzy_pairs:
        print("  Examples (kept → removed, jaccard):")
        for kept, removed, j in fuzzy_pairs[:10]:
            print(f"    {kept}  →  {removed}  ({j:.3f})")

    # --- Pass 3: Contamination check ---
    print("\nPass 3: Train/eval contamination check...", flush=True)
    try:
        eval_rows = load_eval_candidates(args.cache_dir)
        print(f"  Eval rows loaded: {len(eval_rows)}")
        rows_final, contam_pairs = contamination_check(
            rows_after_fuzzy, eval_rows, threshold=args.threshold
        )
        print(f"  Removed: {len(contam_pairs)} contaminated training rows")
        print(f"  Final training candidates: {len(rows_final)}")
        if contam_pairs:
            print("  Examples (train_id → eval_id, jaccard):")
            for train_id, eval_id, j in contam_pairs[:5]:
                print(f"    {train_id}  →  {eval_id}  ({j:.3f})")
    except Exception as e:
        print(f"  WARNING: contamination check failed: {e}")
        rows_final = rows_after_fuzzy

    # --- Summary ---
    print("\n" + "=" * 70)
    print("DEDUP SUMMARY")
    print("=" * 70)
    print(f"  Input rows:           {len(rows)}")
    print(f"  After exact dedup:    {len(rows_after_exact)}  (-{len(exact_pairs)})")
    print(f"  After fuzzy dedup:    {len(rows_after_fuzzy)}  (-{len(fuzzy_pairs)})")
    print(f"  After contamination:  {len(rows_final)}")
    print(f"  Total removed:        {len(rows) - len(rows_final)}")
    print()
    source_counts_final = Counter(r.source for r in rows_final)
    for src, n in sorted(source_counts_final.items()):
        orig = source_counts[src]
        print(f"    {src:<25} {n}  (was {orig})")
    print("=" * 70)

    if not args.report:
        save_parquet(rows_final, args.output)


if __name__ == "__main__":
    main()
