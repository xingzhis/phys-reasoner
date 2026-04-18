"""Build pool v2: Dr.SCI (difficulty >= 0.5) + UGPhysics + PHYSICS + SciBench-RL.

Rule-A composition (one filter rule per source, then include everything):
  - Dr.SCI physics, filtered to difficulty >= 0.5 (tractable band; Spearman rho=0.42
    vs observed 4B TIR pass_fraction on a 2k probe, confirmed 2026-04-18).
  - UGPhysics cleaned corpus (no per-problem difficulty labels, no filter).
  - PHYSICS (desimfj) cleaned corpus (no public leaderboard, self-split ok).
  - SciBench-RL: cleaned corpus (= native train intersect cleaning, 280 rows);
    native test (153 rows, disjoint textbooks: class/diff/matter).

Contamination guards:
  - Cross-source MD5(normalized problem text) intersections must all be 0.
  - SciBench corpus-train intersect native-test must be 0 (verified: they're from
    different textbooks - atkins/stat/fund/... in train, class/diff/matter in test).

Per-source fresh stratified splits (seed=42):
    Dr.SCI >=0.5  (19,409):   train 18,409 | dev 500 | test 500
    UGPhysics     (5,414):    train 5,014  | dev 200 | test 200
    PHYSICS       (793):      train 500    | dev 100 | test 193
    SciBench-RL   (280+153):  train 280    | dev 0   | test 153 (native)
    Combined:                 train 24,203 | dev 800 | test 1,046

Output: data/processed/pool_v2/tir/{train,validation,test}.parquet

To derive CoT variant after: python3 scripts/build_cot_parquets.py \
    --in-dir data/processed/pool_v2/tir --out-dir data/processed/pool_v2/cot

To push: HF_TOKEN=... python3 scripts/push_dataset.py \
    --repo-id xingzhi0/phys-tir --data-dir data/processed/pool_v2/tir
"""
from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.ipc as ipc
import pyarrow.parquet as pq

# Reuse existing helpers from the build_training_parquets module.
sys.path.insert(0, str(Path(__file__).parent))
from build_training_parquets import (  # type: ignore
    _build_prompt,
    _build_reward_model,
    _build_extra_info,
    normalize_primary_answer_type,
    _has_fig_ref,
    map_domain_coarse,
)

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "processed"
OUT_DIR = DATA_DIR / "pool_v2" / "tir"

DRSCI_CLEAN = DATA_DIR / "drsci_physics_clean.parquet"
CORPUS_FILTERED = DATA_DIR / "candidates_filtered.parquet"
SCIBENCH_TEST_ARROW = (
    ROOT / "data" / "hf_cache" / "Sihangli___scibench-rl" / "default" / "0.0.0"
    / "4f8f1cab15a7ff652191451450773c5d917a647f" / "scibench-rl-test.arrow"
)

# Target split sizes per source (dev, test; train is remainder).
DRSCI_DEV, DRSCI_TEST = 500, 500
UG_DEV, UG_TEST = 200, 200
PHYSICS_DEV, PHYSICS_TEST = 100, 193
SCIBENCH_DEV = 0  # per discussion - a 20-row SciBench dev is noise-bound

SEED = 42

# Keep source tags as they appear in candidates_filtered.parquet
# (checked 2026-04-18 on roberts).
UG_SOURCE = "UGPhysics"
PHYSICS_SOURCE = "PHYSICS"
SCIBENCH_SOURCE = "SciBench_RL"

# Union schema for merged extra_info struct (matches merge_splits.py).
UNION_EXTRA_INFO_TYPE = pa.struct([
    ("answer_type", pa.string()),
    ("difficulty", pa.string()),          # stringified on both sides
    ("domain", pa.string()),              # curated only
    ("domain_coarse", pa.string()),       # curated only
    ("from", pa.string()),                # drsci only
    ("primary_answer_type", pa.string()), # curated only
    ("problem", pa.string()),
    ("source", pa.string()),              # curated only
    ("subject", pa.string()),             # drsci only
    ("tolerance", pa.float64()),
    ("unit", pa.string()),
])


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _norm_hash(s: str) -> str:
    s = " ".join(str(s).lower().strip().split())
    return hashlib.md5(s.encode()).hexdigest()


def _assert(cond: bool, msg: str) -> None:
    if not cond:
        raise SystemExit(f"ASSERTION FAILED: {msg}")
    print(f"  OK: {msg}")


def _stratified_split(df: pd.DataFrame, strat_key: str, n_dev: int, n_test: int,
                      seed: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Stratified split by `strat_key` column.

    Allocates dev/test proportionally across strata, remainder is train.
    Tiny strata (<3 rows) stay entirely in train.
    """
    rng = np.random.default_rng(seed)
    total = len(df)
    train_frac = (total - n_dev - n_test) / total
    dev_frac = n_dev / total
    # test_frac = 1 - train_frac - dev_frac

    train_parts: list[pd.DataFrame] = []
    dev_parts: list[pd.DataFrame] = []
    test_parts: list[pd.DataFrame] = []

    for _, grp in df.groupby(strat_key, sort=False):
        n = len(grp)
        if n < 3:
            train_parts.append(grp)
            continue
        perm = rng.permutation(grp.index.to_numpy())
        n_d = max(1, int(round(n * dev_frac)))
        n_t = max(1, int(round(n * (n_test / total))))
        # ensure we leave at least 1 in train
        n_d = min(n_d, n - 2)
        n_t = min(n_t, n - 1 - n_d)
        dev_idx = perm[:n_d]
        test_idx = perm[n_d:n_d + n_t]
        train_idx = perm[n_d + n_t:]
        train_parts.append(grp.loc[train_idx])
        dev_parts.append(grp.loc[dev_idx])
        test_parts.append(grp.loc[test_idx])

    tr = pd.concat(train_parts, ignore_index=True) if train_parts else df.iloc[0:0]
    dv = pd.concat(dev_parts, ignore_index=True) if dev_parts else df.iloc[0:0]
    te = pd.concat(test_parts, ignore_index=True) if test_parts else df.iloc[0:0]
    return tr, dv, te


def _diff_bin(d: float) -> str:
    if d <= 0.5:
        return "0.5"  # lower boundary of kept band
    elif d <= 0.625:
        return "0.625"
    else:
        return "0.75"


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def load_drsci_ge05() -> pd.DataFrame:
    print("\n=== Loading Dr.SCI physics (filter difficulty>=0.5) ===")
    df = pd.read_parquet(DRSCI_CLEAN)
    n_total = len(df)
    print(f"  drsci_physics_clean rows: {n_total}")

    # filter diff >= 0.5
    df = df[df["extra_info.difficulty"] >= 0.5].copy()
    print(f"  after diff>=0.5: {len(df)}")

    # filter unknown answer_type (match process_drsci)
    n_before = len(df)
    df = df[df["inferred_answer_type"] != "unknown"].copy()
    print(f"  after drop unknown answer_type: {len(df)} (dropped {n_before - len(df)})")

    # filter NaN from
    n_before = len(df)
    df = df[df["extra_info.from"].notna()].copy()
    print(f"  after drop NaN from: {len(df)} (dropped {n_before - len(df)})")

    # filter figure refs
    n_before = len(df)
    fig_mask = df["extra_info.question"].astype(str).apply(_has_fig_ref)
    df = df[~fig_mask].copy()
    print(f"  after figure filter: {len(df)} (dropped {n_before - len(df)})")

    return df


def load_corpus_subset(sources: Iterable[str]) -> pd.DataFrame:
    print(f"\n=== Loading corpus (candidates_filtered) subset: {list(sources)} ===")
    df = pd.read_parquet(CORPUS_FILTERED)
    df = df[df["source"].isin(list(sources))].copy()
    print(f"  rows: {len(df)}")
    print(f"  per source: {df['source'].value_counts().to_dict()}")
    return df


def load_scibench_native_test() -> pd.DataFrame:
    """Load native SciBench-RL test (153 rows) and reshape to corpus schema.

    Maps HF arrow columns -> candidates_filtered.parquet schema so the row
    can share the same _build_prompt / _build_extra_info pathway.
    """
    print("\n=== Loading SciBench-RL native test ===")
    with open(SCIBENCH_TEST_ARROW, "rb") as fp:
        df = ipc.RecordBatchStreamReader(fp).read_all().to_pandas()
    print(f"  native test rows: {len(df)}")

    # All native test is physics (class/diff/matter); no chemistry to exclude.
    chemistry_like = df["source"].isin({"atkins", "chemmc"}).sum()
    _assert(chemistry_like == 0,
            f"native SciBench test has no chemistry rows (found {chemistry_like})")

    out = pd.DataFrame({
        "problem_id": [f"SciBench_RL_native_test_{i:05d}" for i in range(len(df))],
        "problem": df["problem_text"].astype(str).str.strip(),
        "answer": df["answer_number"].astype(str).str.strip(),
        "answer_type": "numerical",
        "source": SCIBENCH_SOURCE,
        "difficulty": "",
        "split": "native_test",
        "unit": df["unit"].astype(str).str.strip(),
        "tolerance": 0.05,
        "domain": df["source"].astype(str),  # textbook: class/diff/matter
        "language": "en",
    })
    print(f"  reshaped to corpus schema: {len(out)} rows")
    return out


# ---------------------------------------------------------------------------
# Prompt application
# ---------------------------------------------------------------------------

def apply_drsci_rowbuild(df: pd.DataFrame) -> pd.DataFrame:
    q_col = "extra_info.question"
    at_col = "inferred_answer_type"
    df = df.copy()

    def _diff(val) -> float:
        try:
            return float(val)
        except (ValueError, TypeError):
            return 0.0

    def _prompt(r):
        return _build_prompt(str(r[q_col]), answer_type=r.get(at_col, ""))

    def _rew(r):
        return _build_reward_model(r["reward_model.ground_truth"])

    def _extra(r):
        return _build_extra_info(
            answer_type=r.get(at_col, "numerical"),
            unit="",
            tolerance=0.05,
            problem=str(r[q_col]),
            subject=str(r.get("extra_info.subject", "")),
            difficulty=_diff(r.get("extra_info.difficulty", 0.0)),
            **{"from": str(r.get("extra_info.from", ""))},
        )

    out = pd.DataFrame({
        "data_source": df.get("data_source", pd.Series(["Dr. SCI"] * len(df), index=df.index)),
        "prompt": df.apply(_prompt, axis=1),
        "reward_model": df.apply(_rew, axis=1),
        "extra_info": df.apply(_extra, axis=1),
    })
    return out


def apply_corpus_rowbuild(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["_primary_answer_type"] = df["answer_type"].apply(normalize_primary_answer_type)

    def _safe_tol(v, default=0.05):
        try:
            return float(v) if str(v).strip() else default
        except (ValueError, TypeError):
            return default

    def _prompt(r):
        return _build_prompt(str(r["problem"]), answer_type=r["_primary_answer_type"])

    def _rew(r):
        return _build_reward_model(r["answer"])

    def _extra(r):
        return _build_extra_info(
            answer_type=r.get("answer_type", "numerical"),
            primary_answer_type=normalize_primary_answer_type(r.get("answer_type", "numerical")),
            unit=str(r.get("unit", "")),
            tolerance=_safe_tol(r.get("tolerance", 0.05)),
            problem=str(r.get("problem", "")),
            source=str(r.get("source", "")),
            difficulty=str(r.get("difficulty", "")),
            domain=str(r.get("domain", "")),
            domain_coarse=map_domain_coarse(r.get("domain", "")),
        )

    out = pd.DataFrame({
        "data_source": df.get("source", pd.Series([""] * len(df), index=df.index)),
        "prompt": df.apply(_prompt, axis=1),
        "reward_model": df.apply(_rew, axis=1),
        "extra_info": df.apply(_extra, axis=1),
    })
    return out


# ---------------------------------------------------------------------------
# Merge to union schema
# ---------------------------------------------------------------------------

def _to_union(df: pd.DataFrame, pool_tag: str) -> pa.Table:
    """Promote extra_info struct to UNION_EXTRA_INFO_TYPE and append `pool` col.

    Mirrors merge_splits.reconcile().
    """
    t = pa.Table.from_pandas(df, preserve_index=False)
    ei = t["extra_info"].to_pylist()
    for d in ei:
        if d is None:
            continue
        if d.get("difficulty") is not None:
            d["difficulty"] = str(d["difficulty"])
    new_ei = pa.array(ei, type=UNION_EXTRA_INFO_TYPE)
    idx = t.schema.get_field_index("extra_info")
    t = t.set_column(idx, pa.field("extra_info", UNION_EXTRA_INFO_TYPE), new_ei)
    pool_col = pa.array([pool_tag] * len(df), type=pa.string())
    return t.append_column("pool", pool_col)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--report", action="store_true",
                    help="Dry run: print what would happen, don't write files")
    args = ap.parse_args()

    # ---- load ------------------------------------------------------------
    drsci = load_drsci_ge05()
    corpus = load_corpus_subset([UG_SOURCE, PHYSICS_SOURCE, SCIBENCH_SOURCE])
    ug = corpus[corpus["source"] == UG_SOURCE].reset_index(drop=True).copy()
    physics = corpus[corpus["source"] == PHYSICS_SOURCE].reset_index(drop=True).copy()
    sb_train = corpus[corpus["source"] == SCIBENCH_SOURCE].reset_index(drop=True).copy()
    sb_native_test = load_scibench_native_test()

    # ---- contamination checks -------------------------------------------
    print("\n=== Contamination checks (normalized MD5) ===")
    drsci_hashes = set(drsci["extra_info.question"].apply(_norm_hash))
    ug_hashes = set(ug["problem"].apply(_norm_hash))
    physics_hashes = set(physics["problem"].apply(_norm_hash))
    sb_train_hashes = set(sb_train["problem"].apply(_norm_hash))
    sb_test_hashes = set(sb_native_test["problem"].apply(_norm_hash))

    # Pairwise checks that must be zero.
    checks = [
        ("drsci ∩ ugphysics", drsci_hashes & ug_hashes),
        ("drsci ∩ physics", drsci_hashes & physics_hashes),
        ("drsci ∩ sb_train", drsci_hashes & sb_train_hashes),
        ("drsci ∩ sb_test",  drsci_hashes & sb_test_hashes),
        ("ug ∩ physics",     ug_hashes & physics_hashes),
        ("ug ∩ sb_train",    ug_hashes & sb_train_hashes),
        ("ug ∩ sb_test",     ug_hashes & sb_test_hashes),
        ("physics ∩ sb_train", physics_hashes & sb_train_hashes),
        ("physics ∩ sb_test",  physics_hashes & sb_test_hashes),
        ("sb_train ∩ sb_test", sb_train_hashes & sb_test_hashes),
    ]
    for label, inter in checks:
        _assert(len(inter) == 0, f"{label} = 0 (got {len(inter)})")

    # ---- row-count assertions -------------------------------------------
    print("\n=== Row count sanity ===")
    # Dr.SCI pre-filter >=0.5 is 19,409 (verified); after also dropping unknown
    # answer_type / NaN from / figure refs we land at 18,833.
    _assert(len(drsci) == 18833, f"drsci >=0.5 post-filters == 18,833 (got {len(drsci)})")
    _assert(len(ug) == 5414, f"UGPhysics == 5,414 (got {len(ug)})")
    _assert(len(physics) == 793, f"PHYSICS == 793 (got {len(physics)})")
    _assert(len(sb_train) == 280, f"SciBench corpus == 280 (got {len(sb_train)})")
    _assert(len(sb_native_test) == 153, f"SciBench native test == 153 (got {len(sb_native_test)})")

    # ---- per-source splits -----------------------------------------------
    print("\n=== Per-source stratified splits (seed=42) ===")

    # Dr.SCI: stratify by (from × difficulty_bin × answer_type)
    drsci["_strat"] = (
        drsci["extra_info.from"].astype(str) + "|"
        + drsci["extra_info.difficulty"].apply(_diff_bin) + "|"
        + drsci["inferred_answer_type"].astype(str)
    )
    dr_tr, dr_dv, dr_te = _stratified_split(drsci, "_strat", DRSCI_DEV, DRSCI_TEST, SEED)
    print(f"  Dr.SCI: train={len(dr_tr)}  dev={len(dr_dv)}  test={len(dr_te)}")

    # UGPhysics: stratify by (domain × primary_answer_type)
    ug["_pat"] = ug["answer_type"].apply(normalize_primary_answer_type)
    ug["_strat"] = ug["domain"].astype(str) + "|" + ug["_pat"].astype(str)
    ug_tr, ug_dv, ug_te = _stratified_split(ug, "_strat", UG_DEV, UG_TEST, SEED)
    print(f"  UGPhysics: train={len(ug_tr)}  dev={len(ug_dv)}  test={len(ug_te)}")

    # PHYSICS: stratify by primary_answer_type
    physics["_strat"] = physics["answer_type"].apply(normalize_primary_answer_type)
    ph_tr, ph_dv, ph_te = _stratified_split(physics, "_strat", PHYSICS_DEV, PHYSICS_TEST, SEED)
    print(f"  PHYSICS: train={len(ph_tr)}  dev={len(ph_dv)}  test={len(ph_te)}")

    # SciBench-RL: 280 corpus → all train; 153 native test untouched
    sb_tr = sb_train.copy()
    sb_dv = sb_train.iloc[0:0].copy()
    sb_te = sb_native_test.copy()
    print(f"  SciBench-RL: train={len(sb_tr)}  dev={len(sb_dv)}  test={len(sb_te)} (native)")

    # ---- row-build (TIR prompt + extra_info + reward_model) -------------
    print("\n=== Applying TIR prompt + VeRL schema ===")
    dr_tr_b = apply_drsci_rowbuild(dr_tr)
    dr_dv_b = apply_drsci_rowbuild(dr_dv)
    dr_te_b = apply_drsci_rowbuild(dr_te)
    ug_tr_b = apply_corpus_rowbuild(ug_tr)
    ug_dv_b = apply_corpus_rowbuild(ug_dv)
    ug_te_b = apply_corpus_rowbuild(ug_te)
    ph_tr_b = apply_corpus_rowbuild(ph_tr)
    ph_dv_b = apply_corpus_rowbuild(ph_dv)
    ph_te_b = apply_corpus_rowbuild(ph_te)
    sb_tr_b = apply_corpus_rowbuild(sb_tr)
    sb_te_b = apply_corpus_rowbuild(sb_te)

    # ---- concatenate per split in union schema --------------------------
    print("\n=== Concatenating per split (union schema + pool column) ===")

    def merge_split(parts: list[tuple[pd.DataFrame, str]]) -> pa.Table:
        tabs = [_to_union(df, tag) for df, tag in parts if len(df) > 0]
        return pa.concat_tables(tabs) if tabs else pa.table({})

    train_tab = merge_split([
        (dr_tr_b, "drsci"),
        (ug_tr_b, "curated"),
        (ph_tr_b, "curated"),
        (sb_tr_b, "curated"),
    ])
    val_tab = merge_split([
        (dr_dv_b, "drsci"),
        (ug_dv_b, "curated"),
        (ph_dv_b, "curated"),
        # no scibench dev
    ])
    test_tab = merge_split([
        (dr_te_b, "drsci"),
        (ug_te_b, "curated"),
        (ph_te_b, "curated"),
        (sb_te_b, "curated"),
    ])
    print(f"  train: {len(train_tab)}  val: {len(val_tab)}  test: {len(test_tab)}")

    # ---- composition report ---------------------------------------------
    print("\n=== Composition report ===")
    for name, t in [("train", train_tab), ("val", val_tab), ("test", test_tab)]:
        pool = t["pool"].to_pylist()
        print(f"  {name} pool: drsci={pool.count('drsci')}  curated={pool.count('curated')}")

    # ---- write ----------------------------------------------------------
    if args.report:
        print("\n(--report mode, no files written)")
        return

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pq.write_table(train_tab, str(OUT_DIR / "train.parquet"))
    pq.write_table(val_tab, str(OUT_DIR / "validation.parquet"))
    pq.write_table(test_tab, str(OUT_DIR / "test.parquet"))
    print(f"\nWrote TIR pool parquets to {OUT_DIR}")

    # final read-back verification
    print("\n=== Read-back verification ===")
    for split in ("train", "validation", "test"):
        p = OUT_DIR / f"{split}.parquet"
        df = pd.read_parquet(p)
        print(f"  {split}: {len(df)} rows, cols={list(df.columns)}")
        _assert("prompt" in df.columns, f"{split} has prompt col")
        _assert("reward_model" in df.columns, f"{split} has reward_model col")
        _assert("pool" in df.columns, f"{split} has pool col")

    print("\nNext:")
    print(f"  1. Derive CoT variant:")
    print(f"       python3 scripts/build_cot_parquets.py \\")
    print(f"         --in-dir {OUT_DIR.relative_to(ROOT)} \\")
    print(f"         --out-dir data/processed/pool_v2/cot")
    print(f"  2. Push to HF:")
    print(f"       HF_TOKEN=... python3 scripts/push_dataset.py \\")
    print(f"         --repo-id xingzhi0/phys-tir --data-dir {OUT_DIR.relative_to(ROOT)}")
    print(f"       HF_TOKEN=... python3 scripts/push_dataset.py \\")
    print(f"         --repo-id xingzhi0/phys-cot --data-dir data/processed/pool_v2/cot")


if __name__ == "__main__":
    main()
