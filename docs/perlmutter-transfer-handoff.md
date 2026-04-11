# Perlmutter Transfer — Session Handoff

**Purpose:** hand off context from the 2026-04-10/11 planning session so a fresh Claude session can pick up cold. Paired with `docs/perlmutter-collaborator-questions.md` (the questions doc to send the collaborator) and `.claude/plans/perlmutter-transfer.md` (the full transfer plan, if present).

---

## Situation

- **Deadline:** workshop submission 2026-04-24.
- **Planned Perlmutter bring-up day with collaborator:** 2026-04-13.
- **Production run launch day with collaborator:** 2026-04-16.
- User has **no shell access to Perlmutter** — collaborator runs jobs, user reads streamed logs. This makes every unrehearsed failure expensive.
- Target training: Qwen3.5-4B GRPO, ~1000 steps, ~24h wall-time, async rollout.

## Current progress state

- Zero-shot difficulty probe **in progress** (not finished).
- `scripts/train_async.sh` exists and has been run, but **only single-node 1+1 async**: 1×A100-40G trainer, 1×A100-40G rollout, 1×A6000 xVerify server, all on the small test HPC. **True cross-node async has never run anywhere.**
- `bootstrap_perlmutter.sh` is promised in the questions doc but existence on disk is **unverified** — check before assuming.
- Rollout cost calibration run **in progress** on the small test HPC at time of handoff. User will share numbers.
- GPU-hour estimate for the production run is **not well-justified**, partly because rollout length depends on long token budgets for thinking traces.

## The decision reached this session

**Move the rehearsal to the school HPC.** Reasons:

1. Biggest unmitigated risk before 04-13 is the multi-node async path — NCCL cross-node init, Ray head/worker across nodes, het-job topology, param sync over fabric. None of these are exercised by single-node 1+1.
2. The small test HPC (2×A100-40G no NVLink + 1×4090 + A5000+A6000, slow disk) **cannot** exercise the cross-node code path. It's only good for single-node debugging.
3. Lambda ($400 credit) is wrong for multi-day debugging — treat as last resort.
4. GPU generation doesn't matter for derisking distributed bugs — A100-80G / H100 / H200 all surface the same NCCL/Ray/het-job failures. **Take whatever queues fastest**, including H100/H200 if an A100-80G node is held.

### School HPC specifics (user-confirmed)

- Has 4×A100-80G, 4×H100, 4×H200 nodes with QoS queueing — **hours** of wait, not days, and it's currently weekend so easier to grab.
- **Apptainer + the project SIF are already staged there** — user transferred from school HPC to the current small test HPC earlier, so environment bring-up is a non-issue. Step 0 is "run smoke test," not "build overlay."
- Shared-node allocations are often easier to get than whole nodes. Use shared for correctness debugging; grab a **whole uncontended node** for rollout cost calibration so timing numbers aren't polluted by noisy neighbors.

## Rehearsal checklist for the school HPC

Priority order. Each item derisks a specific Perlmutter failure mode.

1. **2-node async run**, trainer on node A, rollout on node B — proves param sync works cross-node. This is the single highest-value test.
2. **3-group topology**: add the xVerify server as a third group (het-job if supported, otherwise co-locate on rollout node as a fallback). The 3-way het job is what the Perlmutter questions doc is planning for.
3. **Checkpoint → kill → resume end-to-end.** Non-negotiable. 48h wall limit + possible preemption means unresumable runs are disqualified.
4. **Dry-run `bootstrap_perlmutter.sh`** inside the same SIF + overlay. If the script doesn't exist yet, writing and testing it on the school HPC is the right move — same container environment means it'll carry over.
5. **Rollout cost calibration** on the uncontended node. Feeds the GPU-hour estimate the collaborator needs to know before the production run.

## Gaps ranked — discussed this session

### Must fix before 04-13 bring-up
- **Multi-node async validated end-to-end** (items 1–3 above). This is THE gap.
- **`bootstrap_perlmutter.sh` exists and dry-runs clean** (item 4).

### Should fix but cheaper / can be done in parallel
- **Hparams + per-stage steps/epoch justification + training plan.** Real issue but can be locked from the probe data and justified post-hoc once the pipeline runs. Not a bring-up blocker.
- **GPU-hour estimate** — waiting on rollout calibration numbers. Pin down on school HPC if possible.

### Can run in parallel with Perlmutter training (user's own assessment, agreed)
- Eval harness: test set + benchmarks scattered in docs, e.g. **CritPT requires limited-API eval on a private set** — schedule that API quota carefully.
- MATH-500 hard subset as secondary benchmark (already in CLAUDE.md).

## User's resources (for future reference)

| Resource | Specs | Use for |
|---|---|---|
| Small test HPC (current) | 2×A100-40G no NVLink, 1×4090, A5000+A6000, **slow disk** | Single-node debugging, rollout calibration if school HPC unavailable |
| School HPC | 4×A100-80G / 4×H100 / 4×H200 nodes, QoS queue, **SIF+overlay already staged** | **Multi-node rehearsal** (primary recommendation) |
| Lambda | $400 credit | Last resort only — expensive for debugging |
| Perlmutter | via collaborator, no shell access | Production run only, not debugging |

## User's requirements / constraints captured this session

- **No shell access to Perlmutter** — all debugging must happen before transfer.
- Workshop deadline **2026-04-24**.
- Collaborator time windows: **2026-04-13** (bring-up, ~2h) and **2026-04-16** (production launch).
- User will be reading streamed logs during the production run, not driving jobs directly.
- Does not want to waste Lambda credit.
- Wants to pin down GPU-hour estimate before telling collaborator how much budget the run needs.
- Worried about smoothness of transition and unknown-unknowns.

## Open questions for user (asked, not yet resolved at handoff time)

- Rollout calibration numbers (in-flight on small test HPC).
- Whether `bootstrap_perlmutter.sh` actually exists on disk yet.
- Actual queue/partition state on the school HPC once they log in — decides how aggressive the rehearsal schedule can be.

## Items lower-priority than the user initially feared

- **Environment/SIF rebuild on school HPC** — not needed, already staged.
- **Hparams/training plan** — real but solvable post-bring-up, not a transfer blocker.
- **Eval** — can run in parallel with training.

## Items higher-priority than the user initially flagged

- **Checkpoint-resume tested end-to-end** — user didn't mention it, but preemption on a foreign cluster you can't shell into is catastrophic without this.
- **`bootstrap_perlmutter.sh` existence** — the questions doc treats it as done; verify before assuming.

---

**Next session starting point:** user is moving the session to the school HPC. First actions there: (1) check queue and grab nodes, (2) verify SIF+overlay still works with a quick smoke test, (3) attempt item 1 (2-node async run). Then work down the rehearsal checklist.
