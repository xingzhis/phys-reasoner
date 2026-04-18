#!/usr/bin/env bash
# Moved: scripts/dump_rollouts.sh -> eval/inference/rollout.sh
# This stub forwards env vars and args. Update call sites to the new path;
# this stub can be removed once nothing references scripts/dump_rollouts.sh.
exec "$(dirname "$0")/../eval/inference/rollout.sh" "$@"
