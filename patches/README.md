# verl patches

Temporary patches against the `verl/` submodule (fork: `git@github.com:xingzhis/verl.git`, branch `physcode`).
Apply for debugging, revert for production.

Permanent project-specific changes (e.g. think-interrupt) are committed directly to the `physcode` branch
and are always present — no patch needed.

## Usage

```bash
# Apply
cd verl && git apply ../patches/<patch>.patch

# Revert
cd verl && git checkout <file1> <file2>
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
