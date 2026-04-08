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

import json
import logging
from urllib.parse import urlparse

logger = logging.getLogger(__name__)


class XVerifyHTTPClient:
    def __init__(self, url: str, timeout: float = 30.0):
        parsed = urlparse(url)
        if parsed.scheme not in ("http", "https"):
            raise ValueError(f"XVERIFY_URL must be http(s)://host:port/path, got {url!r}")
        self._host = parsed.hostname
        self._port = parsed.port or (443 if parsed.scheme == "https" else 80)
        self._path = parsed.path or "/judge"
        self._scheme = parsed.scheme
        self._timeout = timeout

    def __call__(self, pred_str: str, gold_str: str, problem_str: str = "") -> bool:
        import http.client

        body = json.dumps(
            {"pred": pred_str, "gold": gold_str, "problem": problem_str}
        ).encode("utf-8")
        conn_cls = (
            http.client.HTTPSConnection if self._scheme == "https" else http.client.HTTPConnection
        )
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
                    logger.warning("xverify server returned %s: %s", resp.status, resp.reason)
                    return False
                payload = json.loads(resp.read().decode("utf-8"))
                return bool(payload.get("correct", False))
            finally:
                conn.close()
        except Exception as e:  # noqa: BLE001 — never crash training on judge failure
            logger.warning("xverify call failed (%s): %s", self._host, e)
            return False
