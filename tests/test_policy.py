"""Tests for the Policy module.

Tests that require a real model are skipped when the model path doesn't exist.
"""

from __future__ import annotations

import os
import pytest

MODEL_PATH = "/Users/junhao/models/Qwen3-0.6B"
MODEL_EXISTS = os.path.isdir(MODEL_PATH)

skip_if_no_model = pytest.mark.skipif(
    not MODEL_EXISTS, reason=f"Model not found at {MODEL_PATH}"
)


@skip_if_no_model
def test_policy_instantiation():
    from mlx_agent_rl.core.policy import Policy

    policy = Policy(MODEL_PATH, lora_rank=4, lora_layers=2)
    assert policy.model is not None
    assert policy.tokenizer is not None


@skip_if_no_model
def test_generate_returns_string():
    from mlx_agent_rl.core.policy import Policy

    policy = Policy(MODEL_PATH, lora_rank=4, lora_layers=2)
    policy.eval()
    result = policy.generate("What is 2 + 2?", max_tokens=16)
    assert isinstance(result, str)
    assert len(result) > 0


@skip_if_no_model
def test_generate_with_log_probs_returns_tuple():
    from mlx_agent_rl.core.policy import Policy

    policy = Policy(MODEL_PATH, lora_rank=4, lora_layers=2)
    policy.eval()
    text, log_probs = policy.generate_with_log_probs("What is 2 + 2?", max_tokens=16)
    assert isinstance(text, str)
    assert isinstance(log_probs, list)
    assert len(log_probs) > 0
    # All log-probs should be negative (log of a probability <= 1)
    for lp in log_probs:
        assert isinstance(lp, float)
        assert lp <= 0.0, f"Expected log_prob <= 0, got {lp}"


@skip_if_no_model
def test_generate_with_log_probs_length_matches():
    from mlx_agent_rl.core.policy import Policy

    policy = Policy(MODEL_PATH, lora_rank=4, lora_layers=2)
    policy.eval()
    text, log_probs = policy.generate_with_log_probs("Hello", max_tokens=8)
    # Each token in the decoded text should have one log-prob
    # (we can't easily check exact token count, but length must be > 0)
    assert len(log_probs) > 0


@skip_if_no_model
def test_compute_log_probs_correct_length():
    from mlx_agent_rl.core.policy import Policy

    policy = Policy(MODEL_PATH, lora_rank=4, lora_layers=2)
    policy.eval()

    prompt = "What is 3 + 3?"
    action = " The answer is 6."

    prompt_tokens = list(policy.tokenizer.encode(prompt))
    action_tokens = list(policy.tokenizer.encode(action))

    log_probs = policy.compute_log_probs(prompt_tokens, action_tokens)

    assert isinstance(log_probs, list)
    assert len(log_probs) == len(action_tokens)
    for lp in log_probs:
        assert isinstance(lp, float)
        assert lp <= 0.0, f"Expected log_prob <= 0, got {lp}"


@skip_if_no_model
def test_train_eval_toggle():
    from mlx_agent_rl.core.policy import Policy

    policy = Policy(MODEL_PATH, lora_rank=4, lora_layers=2)
    policy.train()
    policy.eval()


def test_import_policy():
    """Module must be importable even without a model present."""
    from mlx_agent_rl.core import policy  # noqa: F401
