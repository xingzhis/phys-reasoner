"""xVerify LLM fallback judge for physics answer verification."""

from __future__ import annotations

_XVERIFY_PROMPT = """\
You are a diligent and precise assistant tasked with evaluating the correctness of responses.

Question: {problem}
Output sentence: {pred}
Correct answer: {gold}

Determine if the output is correct. Respond with "Correct" or "Incorrect" only.\
"""


class XVerifyJudge:
    """Wraps an xVerify model for answer correctness judgment.

    Args:
        model_name: HuggingFace model ID (default: 0.5B-I).
        device: "cuda" or "cpu".
    """

    def __init__(
        self,
        model_name: str = "IAAR-Shanghai/xVerify-7B-I",
        device: str = "cuda",
    ):
        from huggingface_hub import snapshot_download
        from transformers import AutoModelForCausalLM, AutoTokenizer

        # transformers 5.3.0 _patch_mistral_regex calls model_info() (network)
        # unconditionally when given a Hub ID string. Resolve to actual cache
        # path first so os.path.isdir() returns True and the check is skipped.
        # On a new HPC without cache, pre-stage the model first then run.
        model_path = snapshot_download(model_name, local_files_only=True)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path, dtype="auto", device_map=device
        )
        self.model.eval()
        self._device = device
        # Cache token IDs for "Correct"/"Incorrect" — constant for a given tokenizer
        self._correct_id = self.tokenizer.encode("Correct", add_special_tokens=False)[0]
        self._incorrect_id = self.tokenizer.encode("Incorrect", add_special_tokens=False)[0]

    def _build_prompt(self, pred_str: str, gold_str: str, problem_str: str) -> str:
        return _XVERIFY_PROMPT.format(
            problem=problem_str or "(not provided)",
            pred=pred_str,
            gold=gold_str,
        )

    def __call__(self, pred_str: str, gold_str: str, problem_str: str = "") -> bool:
        """Return True if pred_str is judged correct by xVerify, else False."""
        import torch

        prompt = self._build_prompt(pred_str, gold_str, problem_str)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self._device)
        with torch.no_grad():
            out = self.model.generate(
                **inputs,
                max_new_tokens=10,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        new_tokens = out[0][inputs["input_ids"].shape[1]:]
        response = self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        return response.lower().startswith("correct")

    def get_logprob_score(self, pred_str: str, gold_str: str, problem_str: str = "") -> float:
        """Return P("Correct") from the first token logits (soft score in [0, 1])."""
        import torch
        import torch.nn.functional as F

        prompt = self._build_prompt(pred_str, gold_str, problem_str)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self._device)

        with torch.no_grad():
            logits = self.model(**inputs).logits[0, -1, :]  # last token logits

        pair_logits = logits[[self._correct_id, self._incorrect_id]]
        probs = F.softmax(pair_logits, dim=0)
        return probs[0].item()
