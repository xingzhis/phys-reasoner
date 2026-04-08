# Temporary debug patches

Temporary patches for debugging / inspection runs. Apply for diagnosis,
revert for production. Two scopes:

- `verl_*.patch` — against the `verl/` submodule (fork:
  `git@github.com:xingzhis/verl.git`, branch `physcode`). Apply from
  inside `verl/`.
- `verifier_*.patch` — against this repo's own source tree
  (`src/phys_reasoner/...`). Apply from the repo root.

Permanent project-specific changes (e.g. think-interrupt) are committed directly to the `physcode` branch
and are always present — no patch needed.

## Usage

```bash
# verl patches
cd verl && git apply ../patches/verl_dump_dir.patch
cd verl && git checkout verl/experimental/agent_loop/tool_agent_loop.py verl/trainer/constants_ppo.py

# repo patches (run from repo root)
git apply patches/verifier_dump.patch
git checkout src/phys_reasoner/training/reward.py
```

## Patches

### `verl_dump_dir.patch`

Adds `VERL_DUMP_DIR` support for rollout inspection during smoke tests.

- `verl/experimental/agent_loop/tool_agent_loop.py`: writes two files per rollout to `$VERL_DUMP_DIR`:
  - `rollout_<id>.txt` — decoded token stream (includes special tokens, full trajectory)
  - `rollout_<id>.meta.json` — mask and logprobs metadata:
    - `prompt_len`, `response_len`, `num_trained_tokens`
    - `response_mask` — list of 0/1 (1=trained, 0=injected/tool response)
    - `logprobs_available` — True only when `calculate_log_probs=True`
    - `response_logprobs` — per-token log probs (same length as mask, 0.0 for untrained slots)
    - `logprobs_len_matches_mask` — sanity check: must be True
- `verl/trainer/constants_ppo.py`: propagates `VERL_DUMP_DIR` into Ray worker `runtime_env`.

**When to apply:** smoke tests where you want to inspect raw rollout trajectories.  
**When to revert:** production training — the dump is harmless when `VERL_DUMP_DIR` is unset
(guarded by `if _dump_dir:`), but revert for cleanliness.

### `verifier_dump.patch`

Adds an env-gated per-call dump of the physics verifier (in
`src/phys_reasoner/training/reward.py`) so you can confirm xVerify is
actually being invoked and see what it returned.

- Set `VERIFIER_DUMP_PATH=<file.jsonl>` to enable. One JSON line per
  `compute_score` call:
  - `id`, `pid`, `ts`
  - `data_source`, `answer_type`, `unit`, `gold`, `solution_preview`
  - `xverify_calls` — list of `{pred, gold, result}` (empty list = rule
    short-circuited and xVerify was never invoked; non-empty = xVerify
    was queried, with each call's result)
  - `raw_score` — what `verify_answer` returned (1.0 / 0.0 / -1.0)
- Wraps the singleton xVerify client in a recording proxy *only* when
  `VERIFIER_DUMP_PATH` is set, so production performance is unaffected
  when it isn't.
- Threads in `train_async.sh` already forward `VERIFIER_DUMP_PATH` into
  the apptainer env.

**When to apply:** smoke tests where you want to verify that xVerify is
firing on the right samples and returning sane results.  
**When to revert:** production training — `git checkout
src/phys_reasoner/training/reward.py`.
