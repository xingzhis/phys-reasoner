"""Re-score the diag sample with full xVerify pipeline and print detailed results."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pandas as pd
from phys_reasoner.verifier.router import verify_answer
from phys_reasoner.verifier.xverify_judge import XVerifyJudge

HF_HOME = os.environ.get("HF_HOME", "data/hf_cache")
DIAG_PARQUET = os.path.join(
    os.path.dirname(__file__), "..", "data", "results", "zero_shot_diag.parquet"
)

def main():
    df = pd.read_parquet(DIAG_PARQUET)
    print(f"Loaded {len(df)} samples from diag parquet.\n")

    # Also need problem text — load from the original parquet
    src_path = os.path.join(
        os.path.dirname(__file__), "..", "data", "processed", "candidates_deduped.parquet"
    )
    src = pd.read_parquet(src_path)[["problem_id", "problem"]].set_index("problem_id")

    print("Loading xVerify model (IAAR-Shanghai/xVerify-3B-Ib)...", flush=True)
    judge = XVerifyJudge(model_name="IAAR-Shanghai/xVerify-3B-Ib", device="cuda")
    print("xVerify loaded.\n")

    results = []
    for i, row in df.iterrows():
        problem_text = src.loc[row["problem_id"], "problem"] if row["problem_id"] in src.index else ""

        rule_score = verify_answer(
            row["pred_text"], row["gold_answer"], row["answer_type"],
            xverify_judge=None,
            problem_text=problem_text,
        )
        xv_score = verify_answer(
            row["pred_text"], row["gold_answer"], row["answer_type"],
            xverify_judge=judge,
            problem_text=problem_text,
        )

        print(f"=== Sample {i} | type={row['answer_type']} ===")
        print(f"  GOLD : {repr(row['gold_answer'][:120])}")
        print(f"  PRED : {repr(row['pred_text'][:200])}")
        print(f"  rule-only : {rule_score:+.1f}   with-xVerify : {xv_score:+.1f}")
        if xv_score != rule_score:
            print(f"  *** xVerify changed score: {rule_score} -> {xv_score} ***")
        print()
        results.append({"sample": i, "rule": rule_score, "xverify": xv_score,
                        "answer_type": row["answer_type"]})

    rdf = pd.DataFrame(results)
    print("=== Summary ===")
    print(f"  Rule-only   mean: {rdf['rule'].mean():.3f}")
    print(f"  With-xVerify mean: {rdf['xverify'].mean():.3f}")
    print(f"  xVerify flipped {(rdf['rule'] != rdf['xverify']).sum()} samples")
    print(f"    -1→1: {((rdf['rule']==-1.0) & (rdf['xverify']==1.0)).sum()}  (false neg rescued)")
    print(f"    -1→0: {((rdf['rule']==-1.0) & (rdf['xverify']==0.0)).sum()}  (confirmed wrong)")
    print(f"     0→1: {((rdf['rule']==0.0)  & (rdf['xverify']==1.0)).sum()}  (rule false neg)")


if __name__ == "__main__":
    main()