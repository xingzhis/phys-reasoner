#!/bin/bash
# Project environment setup — source this file once per session.
#
#   source env.sh          # interactive use (from any directory)
#   . env.sh               # equivalent shorthand
#
# All sbatch scripts source this automatically. It sets:
#   ROOT     — absolute path to the repo root
#   SIF      — Apptainer SIF image
#   OVERLAY  — writable overlay image
#   HF_HOME  — HuggingFace model/dataset cache
#   PYTHONNOUSERSITE=1  — prevents ~/.local from shadowing overlay packages
#
# Machine-local overrides go in .env (gitignored, copied from .env.example).
# SIF and OVERLAY can be overridden there if the filenames differ.

# Derive ROOT from this file's location so sourcing from any cwd works.
# Use `pwd -P` (physical path) so apptainer --no-home sees the canonical path
# even if the user enters via a /home/* symlink that isn't mounted in the container.
ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)}"
# Normalize: many sibling scripts set ROOT themselves with plain `pwd` before
# sourcing env.sh, which preserves /home/* symlinks. Re-resolve to the physical
# path so apptainer (--no-home) can always see it.
if [ -d "$ROOT" ]; then
    ROOT="$(cd "$ROOT" && pwd -P)"
fi

# Load machine-local overrides (HF_HOME, SBATCH_PARTITION, SIF, OVERLAY, etc.)
[ -f "$ROOT/.env" ] && source "$ROOT/.env"

# Container images — derive from ROOT unless already overridden in .env
# 017b = fresh overlay on vllm017.latest + transformers==5.3.0 installed on top
# (016.dev.qwen3_5 has vllm 0.1.dev1 which is incompatible with verl >= 0.7.0 requirement)
SIF="${SIF:-$ROOT/verl_vllm017.latest.sif}"
OVERLAY="${OVERLAY:-$ROOT/phys-reasoner-overlay-017.img}"

# HuggingFace cache — local repo cache holds pre-cached models and datasets
HF_HOME="${HF_HOME:-$ROOT/hf_cache}"

# Always isolate from ~/.local to prevent stale package shadowing
PYTHONNOUSERSITE=1

# Unset CUDA_VISIBLE_DEVICES so Ray sees all GPUs and can spread placement groups
# across them. If this is pinned (e.g. to "0"), both rollout and trainer land on
# the same GPU and OOM.
unset CUDA_VISIBLE_DEVICES

export ROOT SIF OVERLAY HF_HOME PYTHONNOUSERSITE

# Export SLURM env vars if set in .env (sbatch reads them as defaults
# when no matching #SBATCH directive is present in the script).
[ -n "${SBATCH_PARTITION:-}" ] && export SBATCH_PARTITION
[ -n "${SBATCH_QOS:-}" ]       && export SBATCH_QOS
[ -n "${SBATCH_ACCOUNT:-}" ]   && export SBATCH_ACCOUNT

WANDB_API_KEY="${WANDB_API_KEY:-}"
export WANDB_API_KEY