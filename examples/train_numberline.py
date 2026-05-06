"""Train a NumberLine agent — text port of verl-agent's gym-cards NumberLine."""

import random
import sys

from mlx_agent_rl.core.trainer import Trainer, TrainerConfig


def make_numberline_dataset(
    n: int, max_position: int = 5, seed: int = 0
) -> list[dict]:
    """Generate ``n`` deterministic (start, goal) pairs as a 'dataset'."""
    rng = random.Random(seed)
    data = []
    for _ in range(n):
        start = rng.randint(0, max_position)
        goal = rng.randint(0, max_position)
        if goal == start:
            goal = (goal + 1) % (max_position + 1)
        data.append(
            {
                # The env parses ``start=S|goal=G`` to recover the exact pair;
                # this also goes into the chat-template's user message.
                "prompt": f"start={start}|goal={goal}",
                "answer": goal,
            }
        )
    return data


def main() -> None:
    config_path = sys.argv[1] if len(sys.argv) > 1 else "configs/numberline.yaml"
    config = TrainerConfig.from_yaml(config_path)
    n_train = int(sys.argv[2]) if len(sys.argv) > 2 else 200
    n_val = int(sys.argv[3]) if len(sys.argv) > 3 else 64

    train = make_numberline_dataset(n_train, seed=0)
    val = make_numberline_dataset(n_val, seed=12345)
    print(f"Loaded {len(train)} train + {len(val)} val episodes", flush=True)
    trainer = Trainer(config=config, dataset=train, val_dataset=val)
    trainer.train()


if __name__ == "__main__":
    main()
