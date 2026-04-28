"""Policy module wrapping mlx-lm for generation and log-prob computation."""

from __future__ import annotations

import contextlib

import mlx.core as mx
import mlx.nn as nn
from mlx_lm import load
from mlx_lm.generate import generate_step
from mlx_lm.sample_utils import make_sampler
from mlx_lm.tokenizer_utils import TokenizerWrapper
from mlx_lm.tuner.lora import LoRALinear
from mlx_lm.tuner.utils import linear_to_lora_layers


class Policy:
    """Wraps an mlx-lm language model with LoRA support.

    Parameters
    ----------
    model_path:
        HF hub id or local directory for the model.
    quantize:
        If set, quantize model weights to this many bits (e.g. 4 or 8).
    lora_rank:
        LoRA rank to apply to transformer layers.
    lora_layers:
        Number of trailing transformer blocks to apply LoRA to.
    temperature:
        Sampling temperature.
    top_p:
        Nucleus-sampling p value.
    """

    def __init__(
        self,
        model_path: str,
        quantize: int | None = None,
        lora_rank: int = 8,
        lora_layers: int = 8,
        temperature: float = 0.7,
        top_p: float = 0.8,
    ) -> None:
        self.temperature = temperature
        self.top_p = top_p

        # Load model + tokenizer
        self.model, self.tokenizer = load(model_path)

        # Optional quantization
        if quantize is not None:
            nn.quantize(self.model, bits=quantize)

        # LoRA mode: freeze base, apply LoRA adapters to trailing layers.
        # Full-param mode: leave all parameters trainable (lora_layers=0).
        if lora_layers > 0:
            self.model.freeze()
            lora_config = {
                "rank": lora_rank,
                "scale": 1.0,
                "dropout": 0.0,
            }
            linear_to_lora_layers(self.model, lora_layers, lora_config)

        # Wrap tokenizer if needed
        if not isinstance(self.tokenizer, TokenizerWrapper):
            self._wrapped_tokenizer = TokenizerWrapper(self.tokenizer)
        else:
            self._wrapped_tokenizer = self.tokenizer

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def generate(self, prompt: str, max_tokens: int = 256) -> str:
        """Generate text from *prompt* and return the response string."""
        from mlx_lm import generate as mlx_generate

        sampler = make_sampler(temp=self.temperature, top_p=self.top_p)
        return mlx_generate(
            self.model,
            self.tokenizer,
            prompt=prompt,
            max_tokens=max_tokens,
            sampler=sampler,
            verbose=False,
        )

    def generate_with_log_probs(
        self, prompt: str, max_tokens: int = 256
    ) -> tuple[str, list[float]]:
        """Generate text and collect per-token log probabilities.

        Returns
        -------
        text:
            The generated string (decoded).
        log_probs:
            List of per-token log-probabilities (one per generated token).
        """
        prompt_tokens = self._wrapped_tokenizer.encode(prompt)
        if isinstance(prompt_tokens, list):
            prompt_array = mx.array(prompt_tokens)
        else:
            prompt_array = prompt_tokens

        sampler = make_sampler(temp=self.temperature, top_p=self.top_p)

        generated_tokens: list[int] = []
        log_probs: list[float] = []

        for token, logprob_vec in generate_step(
            prompt_array,
            self.model,
            max_tokens=max_tokens,
            sampler=sampler,
        ):
            tok_id = int(token)
            if tok_id in self._wrapped_tokenizer.eos_token_ids:
                break
            generated_tokens.append(tok_id)
            # logprob_vec is shape (vocab,) of log-probabilities
            lp = float(logprob_vec[tok_id])
            log_probs.append(lp)
            if len(generated_tokens) >= max_tokens:
                break

        text = self.tokenizer.decode(generated_tokens)
        return text, log_probs

    def compute_log_probs(
        self, prompt_tokens: list[int], action_tokens: list[int]
    ) -> list[float]:
        """Forward pass to compute log-probs of *action_tokens* given *prompt_tokens*.

        The concatenated sequence ``prompt_tokens + action_tokens`` is fed
        through the model.  The returned log-probs correspond to the
        predictions at each action token position (teacher-forced).

        Returns
        -------
        list of float, length == len(action_tokens).
        """
        full_tokens = prompt_tokens + action_tokens
        input_ids = mx.array(full_tokens[:-1])  # (T-1,) — inputs
        target_ids = full_tokens[len(prompt_tokens):]  # action token ids

        logits = self.model(input_ids[None])  # (1, T-1, vocab)
        logits = logits[0]  # (T-1, vocab)

        log_probs_all = nn.log_softmax(logits, axis=-1)  # (T-1, vocab)

        # Positions of action tokens in the output: they start at index
        # len(prompt_tokens)-1 (the model prediction after the last prompt token)
        action_start = len(prompt_tokens) - 1
        result: list[float] = []
        for i, tok in enumerate(target_ids):
            lp = float(log_probs_all[action_start + i, tok])
            result.append(lp)

        mx.eval(log_probs_all)  # ensure computation is done
        return result

    # ------------------------------------------------------------------
    # Train / eval mode helpers
    # ------------------------------------------------------------------

    def train(self) -> None:
        self.model.train()

    def eval(self) -> None:
        self.model.eval()

    # ------------------------------------------------------------------
    # Reference-model context (for KL penalty against frozen base)
    # ------------------------------------------------------------------

    def _iter_lora_modules(self):
        """Yield every LoRALinear module in the policy."""
        stack: list = [self.model]
        while stack:
            mod = stack.pop()
            if isinstance(mod, LoRALinear):
                yield mod
            for child in mod.children().values() if hasattr(mod, "children") else []:
                if isinstance(child, list):
                    stack.extend(c for c in child if isinstance(c, nn.Module))
                elif isinstance(child, nn.Module):
                    stack.append(child)

    @contextlib.contextmanager
    def reference(self):
        """Run forward passes inside this block as if LoRA were absent.

        Each LoRALinear's ``scale`` is temporarily set to 0, so the layer
        returns just the frozen base linear's output. This gives us
        ``log_prob_under_base_model`` without holding a separate model copy
        in memory — the cost is one extra forward pass per sample.
        """
        modules = list(self._iter_lora_modules())
        saved = [(m, m.scale) for m in modules]
        try:
            for m, _ in saved:
                m.scale = 0.0
            yield
        finally:
            for m, s in saved:
                m.scale = s
