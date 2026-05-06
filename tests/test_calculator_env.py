import pytest
from mlx_agent_rl.environments.calculator import CalculatorEnvironment
from mlx_agent_rl.environments.base import Observation


@pytest.fixture
def env():
    return CalculatorEnvironment()


def test_reset(env):
    obs = env.reset("What is 2 + 2?", answer=4)
    assert isinstance(obs, Observation)
    assert "2 + 2" in obs.text
    assert obs.done is False


def test_extract_action_calculate(env):
    output = "I need to compute this. <action>calculate(2 + 2)</action>"
    action = env.extract_action(output)
    assert action == "calculate(2 + 2)"


def test_extract_action_answer(env):
    output = "The answer is <action>answer(4)</action>"
    action = env.extract_action(output)
    assert action == "answer(4)"


def test_extract_action_invalid(env):
    output = "No action tags here."
    assert env.extract_action(output) is None


def test_extract_action_unknown_tag(env):
    output = "<action>unknown_command()</action>"
    assert env.extract_action(output) is None


def test_step_calculate(env):
    env.reset("What is 3 * 7?", answer=21)
    obs, reward, done = env.step("calculate(3 * 7)")
    assert float(obs.text) == 21.0
    assert reward == 0.0
    assert done is False


def test_step_correct_answer(env):
    env.reset("What is 10 / 2?", answer=5)
    obs, reward, done = env.step("answer(5)")
    assert reward == 1.0
    assert done is True
    assert obs.done is True


def test_step_wrong_answer(env):
    env.reset("What is 10 / 2?", answer=5)
    obs, reward, done = env.step("answer(3)")
    assert reward == 0.0
    assert done is True


def test_step_invalid_expression(env):
    env.reset("What is x?", answer=0)
    obs, reward, done = env.step("calculate(import os)")
    assert reward == 0.0
    assert done is False
    assert "Error" in obs.text or "error" in obs.text.lower()


def test_step_calculate_complex(env):
    env.reset("What is (2 + 3) * 4?", answer=20)
    obs, reward, done = env.step("calculate((2 + 3) * 4)")
    assert float(obs.text) == 20.0
    assert done is False
