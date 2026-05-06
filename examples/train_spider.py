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

import json
import sys
from pathlib import Path

from mlx_agent_rl.core.trainer import Trainer, TrainerConfig


def make_dataset(start: int, n: int) -> list[dict]:
    """Each entry pins one deterministic Spider question (by index)."""
    return [{"prompt": "", "answer": i} for i in range(start, start + n)]


def main() -> None:
    config_path = sys.argv[1] if len(sys.argv) > 1 else "configs/spider.yaml"
    config = TrainerConfig.from_yaml(config_path)
    n_train = int(sys.argv[2]) if len(sys.argv) > 2 else 1000
    n_val = int(sys.argv[3]) if len(sys.argv) > 3 else 128
    # Optional 4th arg: JSON file with pre-filtered question indices, e.g.
    # produced by scripts/build_difficulty_filter.py. The file should look
    # like {"indices": [3, 7, 12, ...], ...}. When provided, train uses
    # the first n_train of those indices instead of [0:n_train].
    filter_file = sys.argv[4] if len(sys.argv) > 4 else None

    if filter_file:
        payload = json.loads(Path(filter_file).read_text())
        all_indices = payload["indices"]
        if n_train > len(all_indices):
            print(
                f"[warn] filter has only {len(all_indices)} indices; "
                f"requested n_train={n_train}, will train on all of them",
                flush=True,
            )
            n_train = len(all_indices)
        train = [{"prompt": "", "answer": i} for i in all_indices[:n_train]]
        print(f"[filter] using {filter_file} → {len(train)} questions",
              flush=True)
    else:
        # train pulls from train_spider.json[0:n_train]; val uses a held-out
        # tail slice from the same file so we get an independent signal
        # during training without standing up a second env.
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
