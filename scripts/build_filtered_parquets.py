"""Merge probe fleet outputs, score, filter, and build new TIR+CoT parquets.

Pipeline:
  1. Walk outputs/probe_qwen3_4b/fleet/chunk_*/slot_*/rollouts_chunk_*.parquet
  2. Merge into single rollouts.parquet (~142k rows)
  3. Call score_probe_rollouts.py logic inline (xverify at current.url)
  4. Group by problem_idx → n_correct/n_total
  5. Filter problem_idx where 1 ≤ n_correct ≤ (n_total-1) (informative bucket)
  6. Write data/processed_tir_filtered/data/train.parquet and
           data/processed_cot_filtered/data/train.parquet
     (same prompts as processed_{tir,cot}/data/train.parquet, filtered)

Run after probe fleet completes + xverify server is healthy:
  apptainer exec ... python3 scripts/build_filtered_parquets.py
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import pandas as pd


def find_rollout_parquets(fleet_root: Path) -> list[Path]:
    parquets = sorted(fleet_root.glob("chunk_*/slot_*/rollouts_chunk_*.parquet"))
    if not parquets:
        raise FileNotFoundError(f"No rollouts_chunk_*.parquet under {fleet_root}")
    return parquets


def merge_shards(parquets: list[Path]) -> pd.DataFrame:
    dfs = [pd.read_parquet(p) for p in parquets]
    df = pd.concat(dfs, ignore_index=True)
    return df


def score_df(df: pd.DataFrame, xverify_url: str) -> pd.DataFrame:
    """Score using the same reward function training uses."""
    from phys_reasoner.training.reward import compute_score
    from phys_reasoner.verifier.xverify_client import XVerifyHTTPClient

    client = XVerifyHTTPClient(xverify_url)
    print(f"[score] xverify health check: {client.health_check()}")

    def recon(row):
        parts = []
        p1 = row.get("phase1_text")
        if p1:
            parts.append(str(p1))
        if row.get("interrupted", False):
            p1b = row.get("phase1b_text")
            if p1b:
                parts.append(str(p1b))
        p2 = row.get("phase2_text")
        if p2:
            parts.append(str(p2))
        return "\n".join(parts)

    # Parallelize over HTTP-bound xverify calls. Serial is ~4 rps → hours; threads
    # lift us near GPU-bound throughput of xverify server (~40-80 rps on 7B).
    import os, time
    from concurrent.futures import ThreadPoolExecutor, as_completed
    n_workers = int(os.environ.get("SCORE_WORKERS", "32"))

    def _score_one(idx_row):
        idx, row = idx_row
        try:
            solution_str = recon(row)
            gold = row.get("gold_answer", "")
            extra_info = row.get("extra_info")
            if not isinstance(extra_info, dict):
                extra_info = {}
            s = compute_score(
                solution_str=solution_str,
                ground_truth=str(gold) if gold is not None else "",
                extra_info=extra_info,
                xverify_judge=client,
            )
            return idx, 1 if s > 0 else 0
        except Exception as e:
            return idx, 0

    scores = [0] * len(df)
    done = 0
    t0 = time.time()
    rows = list(df.iterrows())
    print(f"[score] scoring {len(rows)} rollouts with {n_workers} threads")
    with ThreadPoolExecutor(max_workers=n_workers) as ex:
        futures = [ex.submit(_score_one, ir) for ir in rows]
        for fut in as_completed(futures):
            idx, s = fut.result()
            scores[idx] = s
            done += 1
            if done % 2000 == 0 or done == len(rows):
                elapsed = time.time() - t0
                rate = done / elapsed
                eta_min = (len(rows) - done) / rate / 60 if rate > 0 else 0
                n_correct = sum(scores[:done])  # partial tally
                print(f"[score] {done}/{len(rows)}  {rate:.1f} rollouts/s  ETA {eta_min:.1f}min  partial_hit={n_correct}")
    df["correct"] = scores
    return df


def filter_informative(rollouts_df: pd.DataFrame, min_correct: int = 1, max_correct: int = 7) -> set[int]:
    """Return set of problem_idx where n_correct is in [min, max] inclusive."""
    agg = rollouts_df.groupby("problem_idx")["correct"].agg(["sum", "count"]).reset_index()
    n_total = agg["count"].iloc[0] if len(agg) else 0
    print(f"[filter] n_rollouts per prompt: {agg['count'].min()} - {agg['count'].max()}")
    assert (agg["count"] == n_total).all(), "uneven rollouts per prompt"
    kept = agg[(agg["sum"] >= min_correct) & (agg["sum"] <= max_correct)]
    print(f"[filter] kept {len(kept)}/{len(agg)} prompts ({100*len(kept)/len(agg):.1f}%)")
    # Distribution
    dist = agg["sum"].value_counts().sort_index()
    print(f"[filter] n_correct distribution: {dict(dist)}")
    return set(kept["problem_idx"].astype(int).tolist())


def write_filtered_parquet(src: Path, dst: Path, keep_indices: set[int]) -> None:
    df = pd.read_parquet(src)
    # problem_idx in probe output = original position in parquet
    dst_df = df.iloc[sorted(keep_indices)].reset_index(drop=True)
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst_df.to_parquet(dst, index=False)
    print(f"[write] {src.name} ({len(df)}) → {dst} ({len(dst_df)})")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", default="/pscratch/sd/b/bwhou/17-phy-reasoner/phys-reasoner")
    p.add_argument("--fleet", default="outputs/probe_qwen3_4b/fleet")
    p.add_argument("--xverify_url_file", default="outputs/xverify_endpoints/current.url")
    p.add_argument("--skip_score", action="store_true",
                   help="Skip scoring — use correct column already in rollouts parquet")
    p.add_argument("--merged_out", default="outputs/probe_qwen3_4b/rollouts_merged.parquet")
    p.add_argument("--scored_out", default="outputs/probe_qwen3_4b/rollouts_scored.parquet")
    p.add_argument("--min_correct", type=int, default=1)
    p.add_argument("--max_correct", type=int, default=7)
    args = p.parse_args()

    os.chdir(args.root)

    # Step 1-2: merge
    parquets = find_rollout_parquets(Path(args.fleet))
    print(f"[merge] found {len(parquets)} shard parquets")
    merged = merge_shards(parquets)
    print(f"[merge] merged rows: {len(merged)}")
    Path(args.merged_out).parent.mkdir(parents=True, exist_ok=True)
    merged.to_parquet(args.merged_out, index=False)
    print(f"[merge] wrote {args.merged_out}")

    # Step 3: score
    if args.skip_score:
        scored = merged
    else:
        url_path = Path(args.xverify_url_file)
        assert url_path.exists(), f"xverify URL file missing: {url_path}"
        xverify_url = url_path.read_text().splitlines()[0].strip()
        print(f"[score] xverify URL: {xverify_url}")
        scored = score_df(merged, xverify_url)
        scored.to_parquet(args.scored_out, index=False)
        print(f"[score] wrote {args.scored_out}")

    # Step 4-5: filter
    keep = filter_informative(scored, args.min_correct, args.max_correct)

    # Step 6: build filtered TIR + CoT parquets (same indices, different prompt col)
    for mode in ["tir", "cot"]:
        src = Path(f"data/processed_{mode}/data/train.parquet")
        dst = Path(f"data/processed_{mode}_filtered/data/train.parquet")
        write_filtered_parquet(src, dst, keep)
        # Copy validation + test unchanged
        for split in ["validation", "test"]:
            src_s = Path(f"data/processed_{mode}/data/{split}.parquet")
            dst_s = Path(f"data/processed_{mode}_filtered/data/{split}.parquet")
            if src_s.exists():
                dst_s.parent.mkdir(parents=True, exist_ok=True)
                import shutil
                shutil.copy2(src_s, dst_s)
                print(f"[copy] {src_s} → {dst_s}")

    print("\n=== DONE ===")
    print(f"Filtered dataset size: {len(keep)} prompts")
    print(f"TIR: data/processed_tir_filtered/data/train.parquet")
    print(f"CoT: data/processed_cot_filtered/data/train.parquet")
    # 3-epoch step count
    print(f"At batch=128, 3 epochs → {len(keep) * 3 // 128} training steps")


if __name__ == "__main__":
    main()
