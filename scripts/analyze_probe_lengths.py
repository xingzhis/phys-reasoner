"""Analyze 4B probe rollouts' response lengths per problem.

Output: for several candidate p95 thresholds, how many problems survive
(and what their answer-type distribution looks like).
"""
import pyarrow.dataset as ds
import numpy as np

SRC = "/pscratch/sd/b/bwhou/17-phy-reasoner/phys-reasoner/outputs/probe_qwen3_4b_v2_107k/rollouts_scored_trainmatched_107k.parquet"


def main() -> None:
    print(f"[probe-len] loading {SRC}")
    t = ds.dataset(SRC).to_table()
    print(f"[probe-len] columns: {t.column_names}")
    df = t.to_pandas()
    print(f"[probe-len] rows: {len(df)}")
    # Probe is TIR-format (phase1 + phase1b + code + sandbox + phase2).
    # For a CoT-equivalent length estimate, we sum all generated text
    # (phase1_text + phase1b_text + phase2_text), skipping code/sandbox_stdout
    # since those are tool-specific. This gives a reasonable proxy for CoT
    # response length since in CoT mode the model just does one long reasoning
    # chain instead of the two-phase think/answer split.
    text_cols = [c for c in ["phase1_text", "phase1b_text", "phase2_text"] if c in df.columns]
    print(f"[probe-len] text cols: {text_cols}")
    if not text_cols:
        print("[probe-len] ERROR: no text cols")
        return
    df["_chars"] = 0
    for c in text_cols:
        df["_chars"] += df[c].fillna("").str.len()
    length_col = "_chars"

    # need a token-length heuristic if it's char-length.
    # for a rough conversion, assume ~3.5 chars per token; but safer to check
    # via compare to typical max_response_length=2048 tokens.
    max_len = df[length_col].max()
    print(f"[probe-len] max of '{length_col}' = {max_len}")
    median_len = df[length_col].median()
    print(f"[probe-len] median of '{length_col}' = {median_len}")
    if max_len > 2500:
        # Likely characters; convert by /3.5
        print(f"[probe-len] treating '{length_col}' as CHARACTERS (max {max_len} > 2500)")
        df["token_len_est"] = df[length_col] / 3.5
    else:
        # Treat as tokens directly
        print(f"[probe-len] treating '{length_col}' as TOKENS (max {max_len})")
        df["token_len_est"] = df[length_col]

    # per-problem stats
    group_col = None
    for c in ["problem_hash", "problem_idx", "problem_id"]:
        if c in df.columns:
            group_col = c
            break
    if group_col is None:
        print(f"[probe-len] ERROR: no problem grouping col, columns = {df.columns.tolist()}")
        return
    print(f"[probe-len] grouping by: {group_col}")

    extra_cols = {}
    if "answer_type" in df.columns:
        extra_cols["answer_type"] = ("answer_type", "first")

    per_prob = df.groupby(group_col).agg(
        n_rollouts=("token_len_est", "size"),
        len_max=("token_len_est", "max"),
        len_p95=("token_len_est", lambda x: np.percentile(x, 95)),
        len_p50=("token_len_est", "median"),
        **extra_cols,
    ).reset_index()

    print(f"[probe-len] unique problems: {len(per_prob)}")
    print(f"[probe-len] n_rollouts/problem distribution:")
    print(per_prob["n_rollouts"].describe().to_string())

    print("\n[probe-len] per-problem length distribution (tokens, estimated):")
    for stat in ["len_p50", "len_p95", "len_max"]:
        s = per_prob[stat]
        print(f"  {stat}: p10={np.percentile(s,10):.0f} p50={np.percentile(s,50):.0f} p90={np.percentile(s,90):.0f} p99={np.percentile(s,99):.0f} max={s.max():.0f}")

    print("\n[probe-len] Subset sizes for various p95-token thresholds:")
    for thr in [1200, 1400, 1500, 1700, 1800, 2000]:
        kept = per_prob[per_prob.len_p95 < thr]
        pct = 100 * len(kept) / len(per_prob)
        if "answer_type" in kept.columns:
            at_counts = kept["answer_type"].value_counts().to_dict()
            at_str = ", ".join(f"{k}={v}" for k, v in at_counts.items())
        else:
            at_str = "(no answer_type col)"
        print(f"  p95<{thr}: {len(kept)} problems ({pct:.1f}%)   {at_str}")

    # also max-based
    print("\n[probe-len] Subset sizes for various max-token thresholds:")
    for thr in [1700, 1900, 2048, 2200]:
        kept = per_prob[per_prob.len_max < thr]
        pct = 100 * len(kept) / len(per_prob)
        print(f"  max<{thr}: {len(kept)} problems ({pct:.1f}%)")


if __name__ == "__main__":
    main()
