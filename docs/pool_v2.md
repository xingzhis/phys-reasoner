# Training pool v2 — data splits and eval benchmarks

Last updated: 2026-04-18.

Consolidated training pool used for the ICML AI4Physics submission. Replaces the
earlier 108k Dr.SCI+UGPhysics pool under compute-constrained subsampling.

See `docs/training-decisions.md` §"Dataset Sizes (pool v2)" for the full
composition rule and rationale; `docs/eval_plan.md` for the full eval plan.

## Composition

| Source | Filter | Train | Dev | Test |
|---|---|---|---|---|
| Dr.SCI physics | `difficulty ≥ 0.5` | 17,827 | 503 | 503 |
| UGPhysics (EN) | none | 4,980 | 217 | 217 |
| PHYSICS (desimfj) | none, 80/20 self-split | 502 | 100 | 191 |
| SciBench-RL | physics only, native train/test | 280 | 0 | 153 (native) |
| **Total** | | **23,589** | **820** | **1,064** |

Every split stratified (seed=42) by source-appropriate keys; contamination
verified zero across all cross-source problem-text hashes.

## Where to find the parquets

**Local (Roberts / anywhere the build script runs):**
```
data/processed/pool_v2/tir/{train,validation,test}.parquet   # TIR system prompt
data/processed/pool_v2/cot/{train,validation,test}.parquet   # CoT system prompt
```

**HuggingFace (mirrored):**
- https://huggingface.co/datasets/xingzhi0/phys-tir (TIR — for the main TIR-GRPO run)
- https://huggingface.co/datasets/xingzhi0/phys-cot (CoT — for the matched baseline)

Both repos carry identical splits; only the system prompt differs. User messages,
`reward_model`, `extra_info`, and `pool` columns are byte-identical.

**Download on a new machine:**
```bash
source .env   # HF_TOKEN
APT python3 scripts/fetch_dataset.py --repo-id xingzhi0/phys-tir --out-dir data/processed_tir
APT python3 scripts/fetch_dataset.py --repo-id xingzhi0/phys-cot --out-dir data/processed_cot
```
Writes to `data/processed_{tir,cot}/data/{train,validation,test}.parquet` — the
paths that `scripts/train_async.sh` and all Perlmutter production sbatches
already default to.

## Test set (in-distribution, pool v2 test.parquet)

The pool v2 test split has 1,064 rows, one parquet, sourced by:

| Filter | Rows | Use |
|---|---|---|
| `pool == 'drsci'` | 503 | Dr.SCI held-out eval (difficulty ≥ 0.5 band) |
| `data_source == 'UGPhysics'` | 217 | UGPhysics held-out eval |
| `data_source == 'PHYSICS'` | 191 | PHYSICS (desimfj) held-out eval (self-split) |
| `data_source == 'SciBench_RL'` | 153 | SciBench-RL — **official native test split** (disjoint textbooks: class/diff/matter vs train's atkins/stat/fund/thermo/calculus/quan) |

All four subsets scored by `src/phys_reasoner/verifier/router.py` (rule-first,
xVerify-7B fallback on expression types).

## External benchmark datasets (held out entirely)

Not in any pool v2 split. Use for contamination-free external eval.

| Benchmark | Size | HF / source | Official scorer | Local cache |
|---|---|---|---|---|
| OlympiadBench OE_TO physics EN | 236 | `Hothan/OlympiadBench`, config `OE_TO_physics_en_COMP` | OpenBMB `eval/auto_scoring_judge.py` | `data/hf_cache/` via loader |
| PHYBench | 1,000 | `Eureka-Lab/PHYBench` | phybench-official EED (Expression Edit Distance) | `data/hf_cache/` via loader |
| ABench Phy_A + Phy_B | 400 + 400 | `inclusionAI/ABench` (GitHub only) | Official `src/eval.py`, 1% tolerance; Phy_B needs all 4 variants | `data/raw/abench/` CSVs |
| CritPt | 70 | `CritPt-Benchmark/CritPt` | Server API (10 submissions/24h) | `data/hf_cache/` via loader |

## Rebuild

The pool is fully reproducible from `data/processed/drsci_physics_clean.parquet`,
`data/processed/candidates_filtered.parquet`, and the cached native SciBench-RL
test arrow (`data/hf_cache/Sihangli___scibench-rl/...`):

```bash
APT python3 scripts/build_pool_v2.py --report   # dry run
APT python3 scripts/build_pool_v2.py            # writes pool_v2/tir/
APT python3 scripts/build_cot_parquets.py \
  --in-dir data/processed/pool_v2/tir \
  --out-dir data/processed/pool_v2/cot
# Push to HF
HF_TOKEN=... APT python3 scripts/push_dataset.py \
  --repo-id xingzhi0/phys-tir --data-dir data/processed/pool_v2/tir
HF_TOKEN=... APT python3 scripts/push_dataset.py \
  --repo-id xingzhi0/phys-cot --data-dir data/processed/pool_v2/cot
```

(`APT` = the standard `apptainer exec` wrapper; see `scripts/bootstrap_perlmutter.sh` for the exact form.)
