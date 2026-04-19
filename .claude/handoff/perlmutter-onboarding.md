# Perlmutter Claude — Onboarding

You are Claude Code running on NERSC Perlmutter, inside a tmux session the collaborator started using xs272's account. The user (xs272) is watching remotely and can send messages, but cannot directly read files or run commands on this machine. Your job is to stay aligned with the user while doing the hands-on work here.

## Rule zero: ask, don't drift

Every time you would otherwise assume — about a path, a flag value, whether a step is "done enough", whether to proceed to the next sbatch — **ask the user a concrete question in chat first**. Confirmations cost one message; undetected drift costs a 48h sbatch. Propose options as `a/b/c` lists so the user can pick with one word.

Specifically:
- Before any `sbatch`, `scancel`, or `git push`: describe what you're about to do and ask to proceed.
- Before any code edit: state the file path, the specific change, and the reason; ask.
- If a log output doesn't match what you expected: stop, paste the surprising line, ask before continuing.
- If you find yourself writing a long plan instead of executing the day-0 checklist: stop and ask.

## What we're working on

**Research.** PhysCode — TIR-GRPO (main) vs CoT-GRPO (ablation) on Qwen3-4B-Thinking-2507 to compare execution-based reward against LaTeX-verifier-based reward for physics problem solving.

**This session's engineering problem.** A 2026-04-17 Perlmutter smoke of the GRPO pipeline took 27 min/step and OOM'd at step 2 on Qwen3.5-4B. Config work has since landed locally to fix this (model switch to Qwen3-4B-Thinking, Ulysses SP, dynamic batching, etc.). **This session validates that fix on Perlmutter with one smoke sbatch.**

## Staged roadmap (Phase 0 is the only active scope)

| Phase | Goal | Scope for this session |
|---|---|---|
| **0** | **Validate ≤5 min/step with the new config on Perlmutter** | **Active — run one sbatch, report findings, ask user before acting on them** |
| 1 | Interpret timing decomposition, pick next lever | Not in scope — user greenlights after Phase 0 |
| 2 | OOM-ceiling bisect (only if Phase 0 OOMs) | Not in scope |
| 3 | Single-knob isolation tests | Not in scope |
| 4 | Multi-trainer-node (design in `plan-multi-trainer-node.md`) | Not in scope |
| 5 | Production TIR + CoT runs | Not in scope |
| 6 | Offline eval | Not in scope |

When the Phase-0 smoke finishes (pass, fail, or OOM), your job is to **report what happened and wait for the user's direction** — do not proceed to Phase 1 without explicit approval.

## Where things are

- **Repo root:** `/pscratch/sd/b/bwhou/17-phy-reasoner/phys-reasoner` (confirm via `pwd`; ask if different).
- **Branch:** `main` on the parent repo; `physcode` on the `verl` submodule. The old feature branch `switch/qwen3-thinking` has been fast-forward-merged into these and is retained only for forensics.
- **Overlay image:** `phys-reasoner-overlay-017.img` (not `-017b`; that name appears in some older docs).
- **Environment:** see `env.sh`; secrets (`WANDB_API_KEY`, `HF_TOKEN`) live in `.env` (gitignored, already staged by the user). Never push `.env`.
- **Sbatch to submit:** `scripts/perlmutter/smoke_tir_het.sbatch` — 3 het-groups (trainer 1 node × 4 GPU + rollout 5 nodes × 4 GPU + xverify 1 GPU shared). Has `TODO(collab)` markers for account/queue/constraint; you'll resolve these with the collaborator.
- **Training data:** `data/processed_tir/data/{train,validation,test}.parquet`, pulled from the `xingzhi0/phys-tir` HF dataset via `scripts/fetch_dataset.py`. The user has re-filtered and pushed a new revision since this handoff was written, so on arrival you must **re-run the fetch and report row counts in chat for sanity-check**. See `day0-checklist.md` §2b.
- **Target branch for new work:** small config/doc fixes land on parent `main` (and verl `physcode`). For bigger changes — e.g. the Phase-4 multi-trainer-node work — cut a feature branch off `main` (parent) / `physcode` (verl). **Never push to verl's public `main`** — that tracks upstream `verl-project/verl` and must stay a clean mirror.

## Reading order on arrival

Read these once, in this order, then execute the day-0 checklist:

1. `.claude/handoff/day0-checklist.md` — **execute this; it's the first action**
2. `.claude/handoff/collaboration-notes.md` — user preferences + current project framing
3. `.claude/handoff/open-questions.md` — uncertainties you need to resolve with the user or by testing
4. `CLAUDE.md` — environment reference (the "Active model: Qwen3-4B-Thinking-2507" section is canonical)
5. `.claude/session-notes.md` — current state summary
6. `perlmutter_debug/SMOKE_HANDOFF.md` — specific Perlmutter sections (chat template check §2.4, pass criteria §5.2, failure modes §6). Read the 2026-04-19 banner at the top first.
7. `perlmutter_debug/FINDINGS.md` — background on the 12-tier speedup catalogue and why we picked the levers we picked. Reference only; don't reread on every question.

Skip on first pass: `.claude/plans/physcode.md` (too broad for this session), `.claude/plans/data-pipeline.md` (done), `.claude/handoff/plan-multi-trainer-node.md` (Phase 4, later).

## Hard halts — stop and ask before continuing

1. **OOM on step 1** of the smoke, with all Tier-B/C levers on.
2. **Reward mean < 0.05** on step 1 AND no `<tool_call>` emissions in the rollout dump.
3. **Step time > 20 min** at the current config.
4. **Request to run `git push --force`, `git push origin main`, `git reset --hard`, or `scancel` on a job you did not submit.**
5. **Request to modify `.env`, the overlay image (`phys-reasoner-overlay-017.img`), or the SIF.**
6. **Collaborator's working sbatch template differs materially** from `scripts/perlmutter/smoke_tir_het.sbatch` (different overrides, queue, or module loads).
7. **Step 1 reward mean < 0.05 AND rollout dump shows no `<tool_call>` emissions** — do not launch a second run without the user.

## Reporting protocol

- Default: report in chat. Short, factual, one screen.
- Exception: if you've written something long (new doc, plan file, big code change), mention the file path and let the user open it; do not paste its full contents into chat.
- When reporting smoke results, include: each step's `[FullyAsyncTrainer] step=` line (grep it out of `train.log`), median step time over steps 2+, pass/fail against SMOKE_HANDOFF §5.2, and 2–3 proposed next actions as `a/b/c`.
