"""Moved: scripts/dump_rollouts.py -> eval/inference/rollout.py.

This stub forwards to the new location. Update call sites; remove once nothing
references scripts/dump_rollouts.py.
"""
import os
import sys

_here = os.path.dirname(os.path.abspath(__file__))
_target = os.path.normpath(os.path.join(_here, "..", "eval", "inference", "rollout.py"))
os.execv(sys.executable, [sys.executable, _target, *sys.argv[1:]])
