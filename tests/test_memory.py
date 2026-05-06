import pytest
from mlx_agent_rl.memory.memory import SlidingMemory


def test_reset_clears_history():
    mem = SlidingMemory(window_size=3)
    mem.update("obs1", "act1")
    mem.reset()
    assert mem.length == 0
    assert mem.get_context() == ""


def test_single_update():
    mem = SlidingMemory(window_size=3)
    mem.update("hello", "world")
    assert mem.length == 1
    ctx = mem.get_context()
    assert "Observation: hello" in ctx
    assert "Action: world" in ctx


def test_window_truncation():
    mem = SlidingMemory(window_size=2)
    mem.update("obs1", "act1")
    mem.update("obs2", "act2")
    mem.update("obs3", "act3")
    assert mem.length == 3  # full history stored
    ctx = mem.get_context()
    # Only the last 2 entries should appear in context
    assert "obs1" not in ctx
    assert "obs2" in ctx
    assert "obs3" in ctx


def test_format_step_labels():
    mem = SlidingMemory(window_size=3)
    mem.update("first obs", "first act")
    mem.update("second obs", "second act")
    ctx = mem.get_context()
    assert "[Step 1]" in ctx
    assert "[Step 2]" in ctx
    assert "Observation: first obs" in ctx
    assert "Action: first act" in ctx
    assert "Observation: second obs" in ctx
    assert "Action: second act" in ctx


def test_empty_context():
    mem = SlidingMemory()
    assert mem.get_context() == ""


def test_window_size_one():
    mem = SlidingMemory(window_size=1)
    mem.update("obs_first", "act_first")
    mem.update("obs_second", "act_second")
    ctx = mem.get_context()
    assert "obs_first" not in ctx
    assert "act_first" not in ctx
    assert "obs_second" in ctx
    assert "act_second" in ctx
    assert "[Step 1]" in ctx
