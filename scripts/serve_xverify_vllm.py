"""xVerify HTTP server backed by vLLM AsyncLLMEngine — drop-in replacement.

Same /judge and /health endpoints as scripts/serve_xverify.py, same JSON shape,
so xverify_client.py and verl training need NO code changes. Internally uses
vLLM continuous batching across all visible GPUs (TP=auto), so concurrent
HTTP requests share a single GPU forward-pass instead of serializing.

Throughput target on 4 × A100 40GB with xVerify-7B:
  - serial transformers (old): ~5 req/s
  - vLLM TP=4 + 32+ concurrent clients: ~150-400 req/s

Behavior identical to old server because:
  - Same model weights (xVerify-7B-I)
  - Same prompt template (_XVERIFY_PROMPT)
  - Same greedy decoding (temperature=0, max_tokens=10)
  - Same output parser (response.lower().startswith("correct"))

Usage:
  PORT=8765 XVERIFY_MODEL=IAAR-Shanghai/xVerify-7B-I TP_SIZE=4 \\
      python3 scripts/serve_xverify_vllm.py
"""
from __future__ import annotations

import argparse
import logging
import os
import uuid
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, HTTPException
from huggingface_hub import snapshot_download
from pydantic import BaseModel

# Re-use the EXACT prompt template from the transformers-based judge so
# byte-equivalent prompts go to the model.
from phys_reasoner.verifier.xverify_judge import _XVERIFY_PROMPT
from vllm import AsyncEngineArgs, SamplingParams
from vllm.engine.async_llm_engine import AsyncLLMEngine

logger = logging.getLogger("xverify_vllm_server")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")


# Module-level config populated in main() before uvicorn.run; used by lifespan.
_CFG: dict = {}


class JudgeRequest(BaseModel):
    pred: str
    gold: str
    problem: str = ""


class JudgeResponse(BaseModel):
    correct: bool
    score: float  # 1.0 if correct else 0.0 (matches old server)
    raw: str = ""  # raw decoded text — useful for debugging mismatches


def build_prompt(pred: str, gold: str, problem: str) -> str:
    return _XVERIFY_PROMPT.format(
        problem=problem or "(not provided)",
        pred=pred,
        gold=gold,
    )


# Globals populated by lifespan.
_engine: AsyncLLMEngine | None = None
_model_name: str = ""
_sp: SamplingParams = SamplingParams(temperature=0.0, max_tokens=10, top_p=1.0)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize the AsyncLLMEngine inside uvicorn's event loop."""
    global _engine, _model_name
    cfg = _CFG
    _model_name = cfg["model"]
    logger.info("resolving %s from local HF cache (offline)", cfg["model"])
    model_path = snapshot_download(cfg["model"], local_files_only=True)
    logger.info("model_path=%s", model_path)
    logger.info(
        "starting AsyncLLMEngine TP=%d gpu_mem=%.2f max_model_len=%d",
        cfg["tp_size"], cfg["gpu_mem"], cfg["max_model_len"],
    )
    _engine = make_engine(model_path, cfg["tp_size"], cfg["gpu_mem"], cfg["max_model_len"])
    logger.info("engine ready")
    yield
    logger.info("shutting down")


app = FastAPI(lifespan=lifespan)


@app.get("/health")
def health():
    return {
        "status": "ok",
        "ready": _engine is not None,
        "model": _model_name,
        "backend": "vllm-async",
    }


@app.post("/judge", response_model=JudgeResponse)
async def judge(req: JudgeRequest):
    if _engine is None:
        raise HTTPException(503, "engine not initialized")
    prompt = build_prompt(req.pred, req.gold, req.problem)
    request_id = uuid.uuid4().hex
    final_text = ""
    async for output in _engine.generate(prompt, _sp, request_id=request_id):
        final_text = output.outputs[0].text
    text = (final_text or "").strip().lower()
    correct = text.startswith("correct")
    return JudgeResponse(correct=correct, score=1.0 if correct else 0.0, raw=final_text)


def make_engine(model_path: str, tp_size: int, gpu_mem: float, max_model_len: int) -> AsyncLLMEngine:
    args = AsyncEngineArgs(
        model=model_path,
        dtype="auto",
        tensor_parallel_size=tp_size,
        gpu_memory_utilization=gpu_mem,
        max_model_len=max_model_len,
        enforce_eager=True,        # xverify generations are tiny; CUDA graphs add startup latency for no benefit
        enable_prefix_caching=True,  # the prompt prefix is shared across all requests
    )
    return AsyncLLMEngine.from_engine_args(args)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default=os.environ.get("HOST_BIND", "0.0.0.0"))
    ap.add_argument("--port", type=int, default=int(os.environ.get("PORT", "8765")))
    ap.add_argument("--model", default=os.environ.get("XVERIFY_MODEL", "IAAR-Shanghai/xVerify-7B-I"))
    ap.add_argument("--tp_size", type=int, default=int(os.environ.get("TP_SIZE", "1")))
    ap.add_argument("--gpu_mem", type=float, default=float(os.environ.get("GPU_MEM", "0.85")))
    ap.add_argument("--max_model_len", type=int, default=int(os.environ.get("MAX_MODEL_LEN", "4096")))
    args = ap.parse_args()

    # Stash for lifespan to read once uvicorn starts the loop.
    _CFG.update({
        "model": args.model,
        "tp_size": args.tp_size,
        "gpu_mem": args.gpu_mem,
        "max_model_len": args.max_model_len,
    })
    logger.info("uvicorn will bind on %s:%d (engine starts in lifespan)", args.host, args.port)
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
