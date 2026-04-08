"""Tiny stdlib HTTP service that wraps XVerifyJudge for remote use.

POST /judge   {pred, gold, problem}        -> {correct: bool, score: float}
GET  /health                                -> {status: "ok", model: ..., ready: bool}

Designed to be launched on a node with one spare GPU and called by the
training reward function via XVERIFY_URL. Single GPU model, so the call into
xVerify is serialized with a lock; HTTP accepts run in threads so a slow
batch never blocks readiness checks.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

logger = logging.getLogger("xverify_server")

_JUDGE = None
_JUDGE_LOCK = threading.Lock()
_THRESHOLD = 0.5  # P("Correct") cutoff for the soft-score path


class _Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):  # quieter access log
        logger.info("%s - %s", self.address_string(), fmt % args)

    def _send_json(self, status: int, payload: dict) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):  # noqa: N802
        if self.path == "/health":
            self._send_json(
                200,
                {
                    "status": "ok",
                    "ready": _JUDGE is not None,
                    "model": getattr(_JUDGE, "_model_name", None),
                },
            )
            return
        self._send_json(404, {"error": "not found"})

    def do_POST(self):  # noqa: N802
        if self.path != "/judge":
            self._send_json(404, {"error": "not found"})
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            raw = self.rfile.read(length) if length else b""
            req = json.loads(raw.decode("utf-8")) if raw else {}
        except Exception as e:  # noqa: BLE001
            self._send_json(400, {"error": f"bad request: {e}"})
            return

        pred = req.get("pred", "")
        gold = req.get("gold", "")
        problem = req.get("problem", "")
        if _JUDGE is None:
            self._send_json(503, {"error": "judge not ready"})
            return
        try:
            with _JUDGE_LOCK:
                # logprob path: one forward pass, no autoregressive decode
                score = _JUDGE.get_logprob_score(pred, gold, problem)
            self._send_json(200, {"correct": score >= _THRESHOLD, "score": score})
        except Exception as e:  # noqa: BLE001
            logger.exception("judge error")
            self._send_json(500, {"error": str(e)})


def _load_judge(model_name: str, device: str):
    # Local import: heavy deps, only needed when actually serving.
    from phys_reasoner.verifier.xverify_judge import XVerifyJudge

    j = XVerifyJudge(model_name=model_name, device=device)
    j._model_name = model_name  # for /health
    return j


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=int(os.environ.get("XVERIFY_PORT", "8765")))
    ap.add_argument("--model", default=os.environ.get("XVERIFY_MODEL", "IAAR-Shanghai/xVerify-7B-I"))
    ap.add_argument("--device", default=os.environ.get("XVERIFY_DEVICE", "cuda"))
    ap.add_argument("--threshold", type=float, default=0.5)
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    global _JUDGE, _THRESHOLD
    _THRESHOLD = args.threshold
    logger.info("loading %s on %s ...", args.model, args.device)
    _JUDGE = _load_judge(args.model, args.device)
    logger.info("ready: serving on %s:%d (threshold=%.2f)", args.host, args.port, _THRESHOLD)

    server = ThreadingHTTPServer((args.host, args.port), _Handler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        logger.info("shutting down")
        server.shutdown()


if __name__ == "__main__":
    sys.exit(main())
