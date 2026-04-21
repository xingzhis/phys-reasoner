# 3. Method

**Target:** 1.75 pages. Status: empty skeleton.

## 3.1 Single-block TIR trajectory
- Schema (one labeled example).
- Design choices: single-block, budget discipline, `enable_thinking` settings.

## 3.2 Rollout mechanism
- Think-interrupt (ScaleRL-style, budget 12288).
- Sandbox: 30s timeout, subprocess, SymPy/NumPy allowlist.
- Token-level format markers and stop tokens.
- **Figure 2:** rollout flow diagram.

## 3.3 Training algorithm
- Dr. GRPO + DAPO-lite (clip-higher, token-mean, overlong shaping, KL off).
- Reward: binary correctness on final boxed answer via rule + xVerify-7B.
- Compute / FSDP / async rollout layout summary (one paragraph; full table in appendix).

---

<!-- prose goes here -->
