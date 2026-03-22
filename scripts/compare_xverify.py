"""Compare xVerify-3B-Ib vs 7B-I on the 10-sample diag set."""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pandas as pd
from phys_reasoner.verifier.router import verify_answer
from phys_reasoner.verifier.xverify_judge import XVerifyJudge

ROOT    = os.path.join(os.path.dirname(__file__), "..")
DIAG    = os.path.join(ROOT, "data", "results", "zero_shot_diag.parquet")
SRC     = os.path.join(ROOT, "data", "processed", "candidates_deduped.parquet")
HF_HOME = os.environ.get("HF_HOME", os.path.join(ROOT, "hf_cache"))

MODELS = [
    "IAAR-Shanghai/xVerify-3B-Ib",
    "IAAR-Shanghai/xVerify-7B-I",
]

def score_all(df, src, judge, label):
    scores = []
    for _, row in df.iterrows():
        problem = src.loc[row["problem_id"], "problem"] if row["problem_id"] in src.index else ""
        s = verify_answer(row["pred_text"], row["gold_answer"], row["answer_type"],
                          xverify_judge=judge, problem_text=problem)
        scores.append(s)
    return scores

def main():
    df  = pd.read_parquet(DIAG)
    src = pd.read_parquet(SRC)[["problem_id", "problem"]].set_index("problem_id")

    # Rule-only baseline
    rule_scores = score_all(df, src, None, "rule-only")

    all_scores = {"rule-only": rule_scores}

    for model_name in MODELS:
        label = model_name.split("/")[-1]
        print(f"\nLoading {label}...", flush=True)
        judge = XVerifyJudge(model_name=model_name, device="cuda")
        print(f"  loaded.", flush=True)
        scores = score_all(df, src, judge, label)
        all_scores[label] = scores
        del judge  # free GPU memory

    # Print table
    labels = list(all_scores.keys())
    print(f"\n{'i':<3} {'type':<30} " + " ".join(f"{l:>12}" for l in labels))
    print("-" * (3 + 1 + 30 + 1 + 13 * len(labels)))
    for i, row in df.iterrows():
        atype = str(row["answer_type"])[:28]
        vals  = " ".join(f"{all_scores[l][i]:>+12.1f}" for l in labels)
        changed = any(all_scores[l][i] != all_scores[labels[0]][i] for l in labels[1:])
        print(f"{i:<3} {atype:<30} {vals}{'  ←' if changed else ''}")

    print(f"\n{'MEAN':<34} " + " ".join(f"{sum(v)/len(v):>+12.3f}" for v in all_scores.values()))

    # Focus on xVerify false negatives (rule=None → xV=0.0)
    print("\n=== Samples where models disagree ===")
    for i, row in df.iterrows():
        scores_i = {l: all_scores[l][i] for l in labels}
        if len(set(scores_i.values())) > 1:
            print(f"  Sample {i} ({row['answer_type']}): " +
                  "  ".join(f"{l}={v:+.1f}" for l, v in scores_i.items()))

if __name__ == "__main__":
    main()