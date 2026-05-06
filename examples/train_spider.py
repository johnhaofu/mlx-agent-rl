"""Train a Spider SQL agent locally (no sidecar, SQLite is in stdlib).

The dataset for the env is just a list of indices into the configured
split's JSON. ``answer`` in each prompt-dict is the index — env's
``reset`` looks it up. Train uses ``train_spider.json`` (7000); val
should run against ``dev.json`` (1034) by setting ``split: validation``
on a separate val env (TODO once we want a held-out val).

For the first run we sample val from the same train pool to keep the
trainer's existing single-env contract, similar to the HotpotQA
arrangement; later we can plug a held-out env in.
"""

import sys

from mlx_agent_rl.core.trainer import Trainer, TrainerConfig


def make_dataset(start: int, n: int) -> list[dict]:
    """Each entry pins one deterministic Spider question (by index)."""
    return [{"prompt": "", "answer": i} for i in range(start, start + n)]


def main() -> None:
    config_path = sys.argv[1] if len(sys.argv) > 1 else "configs/spider.yaml"
    config = TrainerConfig.from_yaml(config_path)
    n_train = int(sys.argv[2]) if len(sys.argv) > 2 else 1000
    n_val = int(sys.argv[3]) if len(sys.argv) > 3 else 128

    # train pulls from train_spider.json[0:n_train]; val uses a held-out
    # tail slice from the same file so we get an independent signal during
    # training without standing up a second env. Final out-of-distribution
    # eval against ``test.json`` runs separately via
    # scripts/eval_spider_by_hardness.py.
    train = make_dataset(start=0, n=n_train)
    val = make_dataset(start=6000, n=n_val)
    print(
        f"Spider: {len(train)} train + {len(val)} val seeds "
        f"(split={config.environment.split}, data={config.environment.data_dir})",
        flush=True,
    )
    trainer = Trainer(config=config, dataset=train, val_dataset=val)
    trainer.train()


if __name__ == "__main__":
    main()
