"""Download phys-tir parquet splits from the Hub into a local dir.

Usage:
  HF_TOKEN=hf_xxx python scripts/fetch_dataset.py \
    --repo-id <user>/phys-tir \
    --out-dir data/processed_tir

Prints the three absolute parquet paths so the training launcher can
pick them up. Files are bit-identical to the originals; pointing VeRL
at them is equivalent to pointing it at the local merged parquets.
"""
import argparse
import os
from pathlib import Path

from huggingface_hub import snapshot_download

SPLIT_FILES = ["train.parquet", "validation.parquet", "test.parquet"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-id", required=True)
    ap.add_argument("--out-dir", default="data/processed_tir")
    ap.add_argument("--revision", default="main")
    args = ap.parse_args()

    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    local = snapshot_download(
        repo_id=args.repo_id,
        repo_type="dataset",
        revision=args.revision,
        allow_patterns=["data/*.parquet", "README.md"],
        local_dir=str(out_dir),
        token=os.environ.get("HF_TOKEN"),
    )
    data_dir = Path(local) / "data"
    for fname in SPLIT_FILES:
        p = data_dir / fname
        if not p.exists():
            raise SystemExit(f"expected file missing after download: {p}")
        print(p)


if __name__ == "__main__":
    main()
