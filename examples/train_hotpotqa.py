"""Train a HotpotQA agent against the BM25 sidecar.

The sidecar (deployed on a LAN machine; see ``hotpotqa-server/server.py``)
hosts the ``hotpot_qa`` distractor split as parquet files and serves
``/reset`` / ``/step`` over HTTP. Mac runs the MLX trainer; the server
holds the dataset + per-question BM25 index.

Dataset is just a list of ``seed_idx`` integers (the sidecar picks the
question text for that index). Train uses ``train`` split, val uses
``validation``. The split is encoded in ``configs/hotpotqa.yaml``.
"""

import sys

from mlx_agent_rl.core.trainer import Trainer, TrainerConfig


def make_dataset(start: int, n: int) -> list[dict]:
    """Each entry pins one deterministic HotpotQA question (by seed_idx)."""
    return [{"prompt": "", "answer": i} for i in range(start, start + n)]


def main() -> None:
    config_path = sys.argv[1] if len(sys.argv) > 1 else "configs/hotpotqa.yaml"
    config = TrainerConfig.from_yaml(config_path)
    n_train = int(sys.argv[2]) if len(sys.argv) > 2 else 1000
    n_val = int(sys.argv[3]) if len(sys.argv) > 3 else 128

    # Both train and val are sampled from each split's beginning.
    # The split itself is set on the env (config.environment.split).
    train = make_dataset(start=0, n=n_train)
    val = make_dataset(start=0, n=n_val)
    # NB: config.environment.split decides whether seed_idx indexes into the
    # train (90,447) or validation (7,405) parquet. The trainer's
    # _evaluate path doesn't switch split — for a held-out val we'd need a
    # second env instance pointed at split='validation'. For the first
    # smoke we keep split=train so val draws from train too (less proper
    # but lets us measure learning curve quickly).
    print(
        f"HotpotQA: {len(train)} train + {len(val)} val seeds "
        f"(split={config.environment.split}, sidecar={config.environment.base_url})",
        flush=True,
    )
    trainer = Trainer(config=config, dataset=train, val_dataset=val)
    trainer.train()


if __name__ == "__main__":
    main()
