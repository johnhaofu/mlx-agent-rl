import pytest
from mlx_agent_rl.data.trajectory import Step, Trajectory


def make_step(done: bool = False, reward: float = 0.0) -> Step:
    return Step(
        prompt_tokens=[1, 2, 3],
        action_tokens=[4, 5],
        log_probs=[-0.1, -0.2],
        reward=reward,
        done=done,
        anchor_obs="question text",
    )


def test_step_creation():
    step = make_step(done=True, reward=1.0)
    assert step.prompt_tokens == [1, 2, 3]
    assert step.action_tokens == [4, 5]
    assert step.log_probs == [-0.1, -0.2]
    assert step.reward == 1.0
    assert step.done is True
    assert step.anchor_obs == "question text"


def test_trajectory_creation():
    steps = [make_step(), make_step(done=True, reward=1.0)]
    traj = Trajectory(steps=steps, episode_reward=1.0, uid="traj-001")
    assert traj.uid == "traj-001"
    assert traj.episode_reward == 1.0
    assert len(traj.steps) == 2


def test_total_steps():
    steps = [make_step() for _ in range(5)]
    traj = Trajectory(steps=steps, episode_reward=0.0, uid="t1")
    assert traj.total_steps == 5


def test_total_steps_empty():
    traj = Trajectory(steps=[], episode_reward=0.0, uid="t2")
    assert traj.total_steps == 0


def test_succeeded_true():
    steps = [make_step(), make_step(done=True)]
    traj = Trajectory(steps=steps, episode_reward=1.0, uid="t3")
    assert traj.succeeded is True


def test_succeeded_false():
    steps = [make_step(done=False), make_step(done=False)]
    traj = Trajectory(steps=steps, episode_reward=0.0, uid="t4")
    assert traj.succeeded is False


def test_succeeded_empty():
    traj = Trajectory(steps=[], episode_reward=0.0, uid="t5")
    assert traj.succeeded is False
