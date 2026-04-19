"""OlympiadBench OE_TO physics (English, text-only competition) loader.

Source: lscpku/olympiad_bench-official, config OE_TO_physics_en_COMP (236 rows;
text-only, no images). Cached locally under
data/hf_cache/lscpku___olympiad_bench-official/OE_TO_physics_en_COMP/...

Output schema matches rollout.py expectations:
    data_source   : 'OlympiadBench/OE_TO_physics_en_COMP'
    prompt        : [{role: 'user', content: <context + '\n\n' + question>}]
    reward_model  : {ground_truth: <joined boxed answer>, style: 'rule'}
    extra_info    : {problem, answer_type, unit, precision, is_multiple_answer,
                     id, subfield, ...}

reward_model.ground_truth is the gold answer pre-formatted for the official
AutoScoringJudge: each item in `final_answer` is wrapped in \\boxed{...} (after
stripping outer $...$) and concatenated with no separator. The judge's
preprocess() extracts boxed content from both gold and prediction strings and
splits on commas, so this preserves official scoring semantics exactly.

`precision` carries the per-row numerical tolerance (`error` field from the
upstream dataset), as either a float or list-of-floats for multi-answer rows.
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd

DEFAULT_CONFIG = "OE_TO_physics_en_COMP"
DEFAULT_HF_CACHE = "data/hf_cache"
DEFAULT_DATASET_DIR = "lscpku___olympiad_bench-official"
DATA_SOURCE_TAG = "OlympiadBench/OE_TO_physics_en_COMP"


def _find_arrow(hf_cache: str, config: str) -> Path:
    """Locate the cached arrow file. Layout:
    {hf_cache}/{dataset_dir}/{config}/{ver}/{hash}/olympiad_bench-official-train.arrow
    """
    base = Path(hf_cache) / DEFAULT_DATASET_DIR / config
    cands = list(base.glob("*/*/olympiad_bench-official-*.arrow"))
    if not cands:
        raise FileNotFoundError(f"No cached arrow under {base}; run datasets.load_dataset to populate.")
    # OE_TO configs ship a single arrow named *-train.arrow even though it's the only split
    cands.sort()
    return cands[0]


def _strip_dollar(s: str) -> str:
    s = s.strip()
    while s.startswith("$") and s.endswith("$") and len(s) >= 2:
        s = s[1:-1].strip()
    return s


def _build_gt_boxed(final_answer: list[str]) -> str:
    parts = [r"\boxed{" + _strip_dollar(a) + "}" for a in final_answer]
    return "".join(parts)


def _parse_precision(error, n_answers: int) -> list[float]:
    """Parse the `error` field into a list of floats of length n_answers.

    Upstream stores this as a string (e.g. '1e-1' or '0.1,1') but pandas may
    surface a NaN float for missing rows. We always return a list (not a scalar)
    so the resulting parquet column has a single arrow type. The official judge
    accepts either a float or a list, so this keeps semantics intact.
    """
    n = max(1, n_answers)
    default = [1e-8] * n
    if error is None:
        return default
    if isinstance(error, float):
        if error != error:  # NaN
            return default
        val = float(error)
        return [val] * n
    s = str(error).strip()
    if not s:
        return default
    parts = [p.strip() for p in s.split(",")]
    out: list[float] = []
    for p in parts:
        try:
            out.append(float(p))
        except ValueError:
            out.append(1e-8)
    if len(out) == 1:
        return [out[0]] * n
    if len(out) < n:
        out += [out[-1]] * (n - len(out))
    return out[:n]


def _read_arrow(path: Path) -> pd.DataFrame:
    import pyarrow as pa
    import pyarrow.ipc as ipc
    with pa.memory_map(str(path), "r") as src:
        try:
            rdr = ipc.RecordBatchStreamReader(src)
        except Exception:
            src.seek(0)
            rdr = ipc.open_file(src)
        tbl = rdr.read_all()
    return tbl.to_pandas()


def load(
    out_path: str,
    config: str = DEFAULT_CONFIG,
    hf_cache: str = DEFAULT_HF_CACHE,
    text_only: bool = True,
    force: bool = False,
) -> str:
    out = Path(out_path)
    if out.exists() and not force:
        n = len(pd.read_parquet(out))
        print(f"[olympiad:{config}] reuse {out} ({n} rows)")
        return str(out)

    arrow_path = _find_arrow(hf_cache, config)
    src = _read_arrow(arrow_path)
    print(f"[olympiad:{config}] loaded {len(src)} rows from {arrow_path}")

    if text_only and "modality" in src.columns:
        src = src[src["modality"] == "Text-only"].reset_index(drop=True)
        print(f"[olympiad:{config}] {len(src)} text-only rows after filter")

    rows: list[dict] = []
    for _, r in src.iterrows():
        question = str(r["question"])
        context = r.get("context") or ""
        problem = (str(context) + "\n\n" + question).strip() if context else question

        final_answer = list(r["final_answer"]) if r["final_answer"] is not None else []
        gt_boxed = _build_gt_boxed(final_answer)
        precision = _parse_precision(r.get("error"), n_answers=max(1, len(final_answer)))

        is_multi = bool(r.get("is_multiple_answer", False))
        answer_type_raw = str(r.get("answer_type") or "").strip().lower()
        # Map OlympiadBench's labels to our verifier vocabulary (best-effort, mostly
        # informational since we use the official scorer for this benchmark).
        answer_type = {
            "numerical": "numerical",
            "expression": "expression",
            "interval": "expression",
            "equation": "equation",
            "tuple": "expression",
        }.get(answer_type_raw, answer_type_raw or "unknown")

        extra_info = {
            "problem": problem,
            "answer_type": answer_type,
            "unit": str(r.get("unit") or ""),
            "precision": precision,
            "is_multiple_answer": is_multi,
            "final_answer": final_answer,
            "id": int(r["id"]) if pd.notna(r.get("id")) else -1,
            "subfield": str(r.get("subfield") or ""),
            "subject": str(r.get("subject") or ""),
            "language": str(r.get("language") or ""),
            "difficulty": str(r.get("difficulty") or ""),
        }

        rows.append({
            "data_source": DATA_SOURCE_TAG,
            "prompt": [{"role": "user", "content": problem}],
            "reward_model": {"ground_truth": gt_boxed, "style": "rule"},
            "extra_info": extra_info,
            "pool": "external",
        })

    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(out, index=False)
    print(f"[olympiad:{config}] wrote {len(rows)} rows → {out}")
    return str(out)


def main() -> None:
    p = argparse.ArgumentParser(description="Build OlympiadBench OE_TO eval parquet")
    p.add_argument("--config", default=DEFAULT_CONFIG)
    p.add_argument("--hf_cache", default=DEFAULT_HF_CACHE)
    p.add_argument("--out", required=True)
    p.add_argument("--include_multimodal", action="store_true",
                   help="Include rows with images (default: text-only).")
    p.add_argument("--force", action="store_true")
    args = p.parse_args()
    load(out_path=args.out, config=args.config, hf_cache=args.hf_cache,
         text_only=not args.include_multimodal, force=args.force)


if __name__ == "__main__":
    main()
