"""Per-answer-type tool-use rate on in-dist pool_v2 TIR cells."""
import argparse
from pathlib import Path
import pandas as pd

MODEL_SLUG = "Qwen-Qwen3-4B-Thinking-2507"
BENCHMARKS = ["pool_v2_drsci", "pool_v2_physics",
              "pool_v2_ugphysics", "pool_v2_scibench"]
TAGS = ["train", "qwen"]


def _get_answer_type(row_extra, row_reward):
    for src in (row_extra, row_reward):
        if isinstance(src, dict):
            v = src.get("answer_type")
            if v is not None:
                return v
    return None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--outputs-eval", default="outputs/eval")
    p.add_argument("--csv", default="outputs/eval/tool_use_by_type.csv")
    args = p.parse_args()

    root = Path(args.outputs_eval)
    rows = []
    for bench in BENCHMARKS:
        for tag in TAGS:
            cell = root / bench / f"tir__{MODEL_SLUG}__{tag}"
            if not (cell / "rollouts.parquet").exists():
                print(f"[skip] {cell} — no rollouts.parquet")
                continue
            r = pd.read_parquet(cell / "rollouts.parquet")
            s = pd.read_parquet(cell / "scored.parquet")
            r = r.set_index(["problem_idx", "rollout_idx"])
            s = s.set_index(["problem_idx", "rollout_idx"])
            common = r.index.intersection(s.index)
            r, s = r.loc[common], s.loc[common]

            r["called"] = r["code"].apply(
                lambda c: c is not None and not isinstance(c, float))
            r["correct"] = s["correct"].astype(bool)

            if "answer_type" in r.columns:
                atype = r["answer_type"]
            else:
                extra = r["extra_info"] if "extra_info" in r.columns else pd.Series(
                    [None] * len(r), index=r.index)
                reward = r["reward_model"] if "reward_model" in r.columns else pd.Series(
                    [None] * len(r), index=r.index)
                atype = pd.Series(
                    [_get_answer_type(e, w) for e, w in zip(extra, reward)],
                    index=r.index,
                )
            r["answer_type"] = atype

            agg = r.groupby("answer_type", dropna=False).agg(
                n=("called", "size"),
                called_pct=("called", lambda c: 100 * c.mean()),
                overall_pass_pct=("correct", lambda c: 100 * c.mean()),
            )
            agg = agg.reset_index()
            agg["bench"] = bench
            agg["tag"] = tag
            rows.append(agg)

    if not rows:
        print("No cells found.")
        return

    df = pd.concat(rows, ignore_index=True)
    df = df[["bench", "tag", "answer_type", "n", "called_pct", "overall_pass_pct"]]

    for (b, t), g in df.groupby(["bench", "tag"]):
        print(f"\n=== {b} / tir/{t} ===")
        pretty = g[["answer_type", "n", "called_pct", "overall_pass_pct"]].copy()
        pretty["answer_type"] = pretty["answer_type"].astype(str)
        print(pretty.to_string(index=False, float_format=lambda x: f"{x:6.1f}"))

    out_csv = Path(args.csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    print(f"\nWrote {out_csv}")

    none_n = int(df[df["answer_type"].isna()]["n"].sum())
    total_n = int(df["n"].sum())
    if total_n:
        pct = 100 * none_n / total_n
        print(f"Rows with answer_type=None: {none_n}/{total_n} ({pct:.1f}%)")


if __name__ == "__main__":
    main()
