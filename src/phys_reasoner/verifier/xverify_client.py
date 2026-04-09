"""HTTP client for a remote xVerify judge service.

Quacks like ``XVerifyJudge``: instances are callable as
``judge(pred, gold, problem) -> bool`` so they can be passed straight into
``verify_answer`` / ``compute_score`` without any other code changes.

Failure policy: on any network/parse error, return False (i.e. "not judged
correct"). The router treats False as "confirmed wrong"; combined with the
rule verifier's positive short-circuit, this matches today's
``xverify_judge=None`` behavior on outage and never injects spurious 1.0s.
"""

from __future__ import annotations

import http.client
import json
import logging
import random
import socket
import time
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

# Transient errors: retry. These are network/connection-level blips where the
# server is probably fine but the call didn't land. 5xx responses are NOT in
# this set — those are real server-side errors (e.g. GPU OOM) and we fail fast.
_TRANSIENT_ERRORS = (
    ConnectionRefusedError,
    ConnectionResetError,
    BrokenPipeError,
    socket.timeout,
    TimeoutError,
    http.client.RemoteDisconnected,
    http.client.BadStatusLine,
)


class XVerifyHTTPClient:
    def __init__(
        self,
        url: str,
        timeout: float = 30.0,
        max_retries: int = 3,
        backoff_min: float = 0.1,
        backoff_max: float = 0.5,
    ):
        parsed = urlparse(url)
        if parsed.scheme not in ("http", "https"):
            raise ValueError(f"XVERIFY_URL must be http(s)://host:port/path, got {url!r}")
        self._host = parsed.hostname
        self._port = parsed.port or (443 if parsed.scheme == "https" else 80)
        self._path = parsed.path or "/judge"
        self._scheme = parsed.scheme
        self._timeout = timeout
        self._max_retries = max_retries
        self._backoff_min = backoff_min
        self._backoff_max = backoff_max

    def __call__(self, pred_str: str, gold_str: str, problem_str: str = "") -> bool:
        body = json.dumps(
            {"pred": pred_str, "gold": gold_str, "problem": problem_str}
        ).encode("utf-8")
        conn_cls = (
            http.client.HTTPSConnection if self._scheme == "https" else http.client.HTTPConnection
        )

        for attempt in range(1, self._max_retries + 1):
            try:
                conn = conn_cls(self._host, self._port, timeout=self._timeout)
                try:
                    conn.request(
                        "POST",
                        self._path,
                        body=body,
                        headers={"Content-Type": "application/json"},
                    )
                    resp = conn.getresponse()
                    if resp.status != 200:
                        # 5xx / 4xx = real server error. Fail fast, no retry.
                        logger.warning(
                            "xverify server returned %s: %s (host=%s, attempt=%d)",
                            resp.status,
                            resp.reason,
                            self._host,
                            attempt,
                        )
                        return False
                    payload = json.loads(resp.read().decode("utf-8"))
                    return bool(payload.get("correct", False))
                finally:
                    conn.close()
            except _TRANSIENT_ERRORS as e:
                if attempt >= self._max_retries:
                    logger.warning(
                        "xverify call failed after %d attempts (host=%s): %s: %s",
                        attempt,
                        self._host,
                        type(e).__name__,
                        e,
                    )
                    return False
                sleep_s = random.uniform(self._backoff_min, self._backoff_max)
                logger.warning(
                    "xverify transient error (host=%s, attempt=%d/%d, retry in %.2fs): %s: %s",
                    self._host,
                    attempt,
                    self._max_retries,
                    sleep_s,
                    type(e).__name__,
                    e,
                )
                time.sleep(sleep_s)
            except Exception as e:  # noqa: BLE001 — never crash training on judge failure
                # Non-transient (JSON parse error, unexpected exception). Fail fast.
                logger.warning(
                    "xverify call failed (host=%s): %s: %s",
                    self._host,
                    type(e).__name__,
                    e,
                )
                return False

        return False
