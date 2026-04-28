"""Tests for Trainer, TrainerConfig, and related config dataclasses."""

from __future__ import annotations

import os
import tempfile

import pytest
import yaml

from mlx_agent_rl.algorithms.dapo import DAPOEstimator
from mlx_agent_rl.algorithms.dr_grpo import DrGRPOEstimator
from mlx_agent_rl.algorithms.gigpo import GiGPOEstimator
from mlx_agent_rl.algorithms.grpo import GRPOEstimator
from mlx_agent_rl.core.trainer import (
    EnvironmentConfig,
    MemoryConfig,
    ModelConfig,
    RolloutConfig,
    Trainer,
    TrainerConfig,
    TrainingConfig,
)


# ---------------------------------------------------------------------------
# TrainerConfig.from_yaml
# ---------------------------------------------------------------------------

SAMPLE_YAML = """\
model:
  path: /tmp/fake-model
  quantize: 4
  lora:
    rank: 4
    layers: 4
rollout:
  group_size: 2
  max_steps: 3
  max_tokens: 64
training:
  algorithm: dapo
  lr: 0.00001
  epochs: 5
  batch_size: 2
  epsilon: 0.1
  epsilon_high: 0.2
environment:
  type: calculator
  invalid_action_penalty: -0.2
memory:
  window_size: 2
"""


def test_from_yaml_basic():
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write(SAMPLE_YAML)
        tmp_path = f.name

    try:
        cfg = TrainerConfig.from_yaml(tmp_path)
    finally:
        os.unlink(tmp_path)

    assert isinstance(cfg, TrainerConfig)
    assert cfg.model.path == "/tmp/fake-model"
    assert cfg.model.quantize == 4
    assert cfg.model.lora_rank == 4
    assert cfg.model.lora_layers == 4
    assert cfg.rollout.group_size == 2
    assert cfg.rollout.max_steps == 3
    assert cfg.training.algorithm == "dapo"
    assert cfg.training.lr == pytest.approx(1e-5)
    assert cfg.training.epochs == 5
    assert cfg.training.epsilon == pytest.approx(0.1)
    assert cfg.training.epsilon_high == pytest.approx(0.2)
    assert cfg.environment.type == "calculator"
    assert cfg.environment.invalid_action_penalty == pytest.approx(-0.2)
    assert cfg.memory.window_size == 2


def test_from_yaml_defaults_for_missing_keys():
    """Missing sections should fall back to defaults."""
    minimal_yaml = "model:\n  path: /tmp/m\n"
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write(minimal_yaml)
        tmp_path = f.name

    try:
        cfg = TrainerConfig.from_yaml(tmp_path)
    finally:
        os.unlink(tmp_path)

    assert cfg.training.algorithm == "grpo"
    assert cfg.memory.window_size == 3


# ---------------------------------------------------------------------------
# Trainer._create_algorithm
# ---------------------------------------------------------------------------

def test_create_algorithm_grpo():
    estimator = Trainer._create_algorithm("grpo")
    assert isinstance(estimator, GRPOEstimator)
    assert not isinstance(estimator, DrGRPOEstimator)  # subclass check


def test_create_algorithm_dr_grpo():
    estimator = Trainer._create_algorithm("dr_grpo")
    assert isinstance(estimator, DrGRPOEstimator)


def test_create_algorithm_dapo():
    estimator = Trainer._create_algorithm("dapo", epsilon=0.1, epsilon_high=0.2)
    assert isinstance(estimator, DAPOEstimator)
    assert estimator.epsilon_high == pytest.approx(0.2)


def test_create_algorithm_gigpo():
    estimator = Trainer._create_algorithm("gigpo")
    assert isinstance(estimator, GiGPOEstimator)


def test_create_algorithm_unknown():
    with pytest.raises(ValueError, match="Unknown algorithm"):
        Trainer._create_algorithm("nonexistent")


# ---------------------------------------------------------------------------
# Trainer._create_environment
# ---------------------------------------------------------------------------

def test_create_environment_calculator():
    from mlx_agent_rl.environments.calculator import CalculatorEnvironment

    env_cfg = EnvironmentConfig(type="calculator")
    env = Trainer._create_environment(env_cfg)
    assert isinstance(env, CalculatorEnvironment)


def test_create_environment_unknown():
    env_cfg = EnvironmentConfig(type="unknown_env")
    with pytest.raises(ValueError, match="Unknown environment type"):
        Trainer._create_environment(env_cfg)


# ---------------------------------------------------------------------------
# Config dataclass instantiation
# ---------------------------------------------------------------------------

def test_model_config_defaults():
    cfg = ModelConfig()
    assert cfg.lora_rank == 8
    assert cfg.lora_layers == 8
    assert cfg.quantize is None


def test_training_config_defaults():
    cfg = TrainingConfig()
    assert cfg.algorithm == "grpo"
    assert cfg.epochs == 10
