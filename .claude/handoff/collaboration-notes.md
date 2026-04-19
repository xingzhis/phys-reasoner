# Collaboration notes — user preferences + project framing

These are observations carried over from prior sessions to save you from re-asking. Verify anything that looks stale; memory can be wrong.

## User

- Researcher working on physics reasoning + LLM training. Comfortable with GRPO/VeRL internals, HuggingFace stack, Apptainer/Slurm.
- Iterative collaborator: proposes direction → discusses → executes. Makes high-level calls (model choice, scope); expects you to handle implementation planning.
- Prefers concise responses. No trailing summaries ("I can read the diff").
- Has explicitly asked that you **not make assumptions on ambiguous design choices** — ask instead.
- This session's special note: user is remote and cannot directly run commands, so is leaning harder on you to ask before acting.

## Project (current as of 2026-04-19)

- **TIR-GRPO vs CoT-GRPO ablation** on Qwen3-4B-Thinking-2507. Same base model, same data, same budgets; only reward signal differs (tool-execution correctness vs LaTeX-verifier).
- **Debug model** (local smokes only): Qwen3-0.6B.
- **SFT is skipped** per the 2026-04-14 Stage-0 probe (22% zero-shot hit rate, above the 15% threshold).
- **Training data:** Dr. SCI (~102k) + UGPhysics (~5.4k). **Held out from training** (do not reintroduce): PHYSICS, OlympiadBench OE_TO, SciBench-RL, PHYBench, ABench, CritPt — these are eval-only.
- **Budgets:** `thinking=12288`, `tool_call=2048`, `tool_response_max=512`, `answer=4096`, total `response_length=18944`. CoT absorbs the tool share into post-interrupt answer budget → same total length, fair ablation.

## Things to ask (don't guess)

- File paths on Perlmutter that differ from CLAUDE.md's Roberts-cluster paths.
- Which sbatch queue/account/constraint to use (the TODO(collab) markers).
- Whether a given lever is worth a separate sbatch or should be bundled with another.
- Any success criterion tighter or looser than what SMOKE_HANDOFF §5.2 states.

## Things NOT to do

- Do not push to `main`. Work on `switch/qwen3-thinking` sub-branches.
- Do not modify `.env` or the overlay image.
- Do not `scancel` any job you did not submit.
- Do not `git push --force` anything without asking.
- Do not reintroduce the 6 held-out benchmarks to training data.
- Do not port the old Qwen3.5-specific overrides (SDPA, `Qwen3_5DecoderLayer` wrap policy, `qwen3_coder` format) unless the user explicitly decides to revert the Qwen3 switch. Historical copy lives in the "If reverting" footnote of CLAUDE.md.
