# Day-0 Checklist

Work through this top-to-bottom on arrival. Every "→ ask user if …" is a hard pause: stop, send the question in chat, wait for an answer. Do not batch questions — one at a time, so the user can respond quickly.

## 1. Orient

- [ ] `pwd` → expect `/pscratch/sd/b/bwhou/17-phy-reasoner/phys-reasoner`. **→ ask user if different.**
- [ ] **Pull latest on both repos before doing anything else**:
  ```bash
  git checkout main && git pull --ff-only origin main
  git submodule update --init --recursive --remote=false   # checkout verl at the SHA parent pins, then we advance verl deliberately below
  git -C verl fetch origin
  git -C verl checkout physcode && git -C verl pull --ff-only origin physcode
  ```
  If either `pull` is not a fast-forward, **→ HALT and ask user** — something diverged that shouldn't have.
- [ ] `git branch --show-current` on parent → expect `main`. `git log --oneline -10` → confirm log contains `d3c8814 handoff package for remote-hosted Perlmutter Claude session` and later post-merge doc commits. **→ ask user if missing.**
- [ ] `git -C verl branch --show-current` → expect `physcode`. `git -C verl log --oneline -5` → confirm HEAD contains `d822b76a fix double timing_s/ prefix on param_sync and merge_val timers`. **→ ask user if missing.**
- [ ] `git status` + `git -C verl status` both clean. If not, **→ ask user before staging or discarding anything.**

## 2. Environment sanity

- [ ] `source env.sh` succeeds; `echo "$SIF $OVERLAY $HF_HOME $ROOT"` shows all four paths.
- [ ] `ls "$SIF" "$OVERLAY"` both exist. Overlay name should be `phys-reasoner-overlay-017.img` (not `-017b`). **→ ask user if different.**
- [ ] `test -f .env && grep -c WANDB_API_KEY .env && grep -c HF_TOKEN .env` → both should be 1. **→ ask user if either is missing; do not create `.env` yourself.**
- [ ] `ls "$HF_HOME/hub"` contains `models--Qwen--Qwen3-4B-Thinking-2507`. If not, **→ ask user** (download command is in SMOKE_HANDOFF §2.3 but don't run without approval, it's ~8GB).

## 2b. Refresh the HF dataset (new filter was pushed to `xingzhi0/phys-tir`)

The user has re-filtered the training data since the parquets last synced. Re-pull from HF before launching anything:

```bash
source env.sh
set -a && source .env && set +a   # brings HF_TOKEN into scope without echoing
PYTHONNOUSERSITE=1 apptainer exec \
  --overlay "$OVERLAY:ro" --no-home --bind /etc/pki:/etc/pki \
  --env "PYTHONNOUSERSITE=1" --env "PYTHONPATH=/opt/phys-extras/" \
  --env "HF_TOKEN=$HF_TOKEN" --env "HF_HOME=$HF_HOME" \
  "$SIF" \
  python3 scripts/fetch_dataset.py --repo-id xingzhi0/phys-tir --out-dir data/processed_tir
```

Then print row counts and **report them in chat for the user to sanity-check**:

```bash
PYTHONNOUSERSITE=1 apptainer exec \
  --overlay "$OVERLAY:ro" --no-home --bind /etc/pki:/etc/pki \
  --env "PYTHONNOUSERSITE=1" --env "PYTHONPATH=/opt/phys-extras/" \
  "$SIF" \
  python3 -c "
import pandas as pd
for split in ['train', 'validation', 'test']:
    df = pd.read_parquet(f'data/processed_tir/data/{split}.parquet')
    print(f'{split}: {len(df):,} rows')
"
```

**Do not proceed past this step until the user confirms the row counts look right.** The old filter produced train=107,977 / val=2,200 / test=2,200 — expect different numbers now; that's the point of this step.

If `fetch_dataset.py` fails with an HF auth error, the HF_TOKEN in `.env` may be stale — **→ ask user**, do not try to regenerate.

## 3. Syntax checks (cheap, catch bit-rot)

- [ ] `bash -n scripts/perlmutter/smoke_tir_het.sbatch` → silent exit.
- [ ] `bash -n scripts/perlmutter/_ray_bringup.sh` → silent exit.
- [ ] `bash -n scripts/train_async.sh` → silent exit.

If any fail: **→ ask user** before editing.

## 4. TODO(collab) resolution

- [ ] `grep -n 'TODO(collab)' scripts/perlmutter/*.sbatch` — lists the account/queue/constraint placeholders.
- [ ] Ask the collaborator (in-person or via the tmux session) for their most recent working sbatch on Perlmutter that uses the same account/queue shape. Read it, note the values.
- [ ] **→ ask user** to confirm the resolved values before editing the sbatch. Do not commit the resolution until approved.

## 5. Chat template probe (confirms hermes format)

Run the short python snippet in `perlmutter_debug/SMOKE_HANDOFF.md §2.4` inside apptainer. Expect the rendered prompt to contain `<think>` and `<tool_call>` with JSON (not XML `<function=…><parameter=…>`).

- [ ] If output matches hermes → proceed.
- [ ] If output is XML (`<function=`) → **HALT. → ask user.** The smoke's `multi_turn.format=hermes` setting will be wrong for this chat template.

## 6. One final pause before sbatch

Summarize in chat:
1. Branch SHAs on both repos (after §1 pull)
2. Environment sanity pass/fail (§2)
3. HF dataset row counts — user confirmed? (§2b)
4. Resolved TODO(collab) values (§4)
5. Chat template result (§5)

Then **ask**: "Ready to submit `scripts/perlmutter/smoke_tir_het.sbatch`?" Wait for explicit approval. Do not submit until the user says yes.

## 7. After submission

- Monitor with `squeue -u $USER`; note the job ID(s).
- Tail `logs/<jobname>_<jobid>.log` — watch for the critical lines listed in `SMOKE_HANDOFF §5.1` (Ray ready, Ulysses SP active, attn impl not sdpa, trainer pinned).
- First step lands → grep `[FullyAsyncTrainer] step=` from the log and paste to chat with a one-sentence read.
- If any of the hard-halt conditions in `perlmutter-onboarding.md` fires → `scancel $JOB` (only the job you submitted), report to chat, wait.

## 8. On completion

Run `scripts/perlmutter/check_smoke.sh` (see file for usage). Report its output in chat plus 2–3 `a/b/c` proposed next actions. **Wait for user direction before starting Phase 1.**
