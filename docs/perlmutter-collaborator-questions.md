# Questions for Perlmutter Collaborator

**Context:** Setting up async GRPO training (Qwen3.5-4B, ~1000 steps, ~24h wall-time) on Perlmutter for a workshop submission due 2026-04-24. See `.claude/plans/perlmutter-transfer.md` for the full plan.

**Send all of these in one message.**

---

## 1. Wall-time limits per QoS

**Resolved from https://docs.nersc.gov/jobs/policy/ — max 48h. Will design for checkpoint-and-resume.**

Still want to confirm with collaborator:
- What's the actual sweet-spot wall-time for fastest backfill scheduling on the GPU partition? (e.g., requesting 12h vs 24h vs 48h — which queues fastest in practice?)
- Any QoS-specific node-hour costs I should know about (e.g., `regular` vs `premium` cost multiplier)?

## 2. Node availability and queue times

- 4×A100-80G nodes: typical queue wait? worst case?
- 4×A100-40G nodes: same question
- Can I request a **heterogeneous job** (one 80G node + one 40G node in the same allocation), e.g.
  ```
  sbatch --het-group=0 --gres=gpu:4 -N1 \
         --het-group=1 --gres=gpu:1 -N1 \
         run.sbatch
  ```
  This would let me put trainer+rollout on the 80G node and the xVerify-7B reward server on the 40G node. If het jobs aren't supported, I'll fall back to co-locating xVerify on the same 80G node.

## 3. Outbound network from compute nodes

- Can compute nodes reach `api.wandb.ai` (port 443)? I want to use WandB for logging.
- Can compute nodes reach `huggingface.co` for one-time model downloads, or do I need to pre-stage everything to `$SCRATCH` from a login node?

## 4. Storage

- Recommended path for ~200GB of model weights + parquets + checkpoints? `$SCRATCH`? `$PSCRATCH`? `$CFS`?
- Is the recommended scratch purged on a timer? If so, what's the policy (e.g., 12-week purge)?
- Apptainer/Singularity SIF images: any size limit, recommended location?

## 5. Apptainer / Singularity

- Which version is installed on compute nodes?
- Are overlays (`apptainer exec --overlay foo.img ...`) supported on compute nodes? On RHEL 8 I've hit issues where OverlayFS whiteouts are silently ignored when the overlay is mounted `:ro`, causing pip-upgraded packages to be invisible at runtime. Has anyone hit this on Perlmutter?
- Is `--nv` the correct GPU passthrough flag, or does Perlmutter have its own convention?
- Any restrictions on bind mounts I should know about?

## 6. Project budget

- How many node-hours do we have for this experiment?
- Any constraint on parallel jobs? (I might want to run an LR sweep with 2 concurrent jobs.)

## 7. SLURM specifics

- Account / partition / qos to submit under
- Whether job preemption is common (affects how often I checkpoint)
- Any cluster-specific environment modules I should `module load` before `apptainer exec`?

## 8. Practical logistics

- Can you sit with me for ~2 hours on **2026-04-13** to do the initial bring-up (run `bootstrap_perlmutter.sh`, smoke test, 1-step 4B test)?
- And again on **2026-04-16** for the production run launch?
- Between those days, are you OK running the training jobs and forwarding logs? I can't get shell access myself, so I'll be reading logs you stream back.

---

## What I'm bringing

- A self-contained `scripts/bootstrap_perlmutter.sh` that builds the environment from the SIF
- The Apptainer SIF (`verl_vllm017.latest.sif`)
- A `run_grpo_perlmutter.sbatch` template (will adapt to your answers above)
- Pre-built training parquets (~150MB) and the training code (`git pull` from a tagged commit)
- A test suite that should pass before we attempt the real training

Total disk: ~50GB SIF + ~200GB weights + ~1GB code/parquets = ~250GB.
