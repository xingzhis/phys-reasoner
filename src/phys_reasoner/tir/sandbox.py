"""Sandboxed code execution for TIR trajectories.

Runs untrusted model-generated Python in a subprocess with:
- Import whitelist (AST-level, pre-subprocess)
- 30s timeout
- 2 GB memory limit (via resource.setrlimit where available)
- stdout truncation
"""

from __future__ import annotations

import ast
import subprocess
import sys
from dataclasses import dataclass, field

ALLOWED_PACKAGES: frozenset[str] = frozenset({
    "numpy", "scipy", "sympy", "pint",
    "math", "cmath", "statistics", "fractions", "decimal",
    "itertools", "functools", "collections", "random",
    "re", "json", "io", "typing", "time",
})

_TRUNCATION_MARKER = "...[truncated]"


@dataclass
class SandboxResult:
    stdout: str = ""
    stderr: str = ""
    timed_out: bool = False
    error: bool = False


def execute_code(
    code: str,
    timeout: float = 30.0,
    max_output_bytes: int = 4096,
) -> SandboxResult:
    """Run code in a subprocess. Never raises. Returns SandboxResult."""
    if not isinstance(code, str):
        return SandboxResult(stderr=f"TypeError: code must be str, got {type(code).__name__}", error=True)

    # --- Import whitelist check (no subprocess needed for fast rejection) ---
    violation = _check_imports(code)
    if violation is not None:
        if violation == "<syntax error>":
            return SandboxResult(stderr="SyntaxError: code could not be parsed", error=True)
        return SandboxResult(
            stderr=f"ImportError: package '{violation}' not in ALLOWED_PACKAGES",
            error=True,
        )

    # --- Memory limit preexec (best-effort; skip if resource unavailable) ---
    def _set_mem_limit():
        try:
            import resource  # noqa: PLC0415
            limit = 2 * 1024 ** 3  # 2 GB
            resource.setrlimit(resource.RLIMIT_AS, (limit, limit))
        except Exception:
            pass

    # --- Subprocess ---
    try:
        proc = subprocess.Popen(
            [sys.executable, "-c", code],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            preexec_fn=_set_mem_limit,
        )
        try:
            raw_out, raw_err = proc.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.communicate()
            return SandboxResult(timed_out=True, error=True)
    except Exception as exc:
        return SandboxResult(stderr=str(exc)[:500], error=True)

    stdout = raw_out.decode("utf-8", errors="replace").strip()
    stderr = raw_err.decode("utf-8", errors="replace")[:500]

    # Truncate stdout
    if len(stdout.encode()) > max_output_bytes:
        truncated = stdout.encode()[:max_output_bytes].decode("utf-8", errors="replace")
        stdout = truncated + _TRUNCATION_MARKER

    error = proc.returncode != 0
    return SandboxResult(stdout=stdout, stderr=stderr, error=error)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _check_imports(code: str) -> str | None:
    """Return the first disallowed top-level package name, or None if all OK.

    Also rejects if the code cannot be parsed (syntax error).
    """
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return "<syntax error>"

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                pkg = alias.name.split(".")[0]
                if pkg not in ALLOWED_PACKAGES:
                    return pkg
        elif isinstance(node, ast.ImportFrom):
            if node.module is None:
                continue
            pkg = node.module.split(".")[0]
            if pkg not in ALLOWED_PACKAGES:
                return pkg
    return None
