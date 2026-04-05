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
ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"

# Load machine-local overrides (HF_HOME, SBATCH_PARTITION, SIF, OVERLAY, etc.)
[ -f "$ROOT/.env" ] && source "$ROOT/.env"

# Container images — derive from ROOT unless already overridden in .env
# 017b = fresh overlay on vllm017.latest + transformers==5.3.0 installed on top
# (016.dev.qwen3_5 has vllm 0.1.dev1 which is incompatible with verl >= 0.7.0 requirement)
SIF="${SIF:-$ROOT/verl_vllm017.latest.sif}"
OVERLAY="${OVERLAY:-$ROOT/phys-reasoner-overlay-017b.img}"

# HuggingFace cache — local repo cache holds pre-cached models and datasets
HF_HOME="${HF_HOME:-$ROOT/hf_cache}"

# Always isolate from ~/.local to prevent stale package shadowing
PYTHONNOUSERSITE=1

export ROOT SIF OVERLAY HF_HOME PYTHONNOUSERSITE
