"""Train a WebShop agent against a remote text-mode HTTP sidecar.

The sidecar is expected to wrap verl-agent's WebAgentTextEnv (see
``deploy/webshop_text_api.py`` for the server-side script). Mac runs the
MLX trainer, all heavy WebShop infra (Flask + 5GB DB + faiss + spaCy lg)
stays on the LAN-attached server.

Dataset is just a list of goal indices: WebShop's own server picks the
instruction text per index. The base config uses goal_idx range
[500, len(goals)) for train and [0, 500) for val to mirror verl-agent's
split.
"""

import sys

from mlx_agent_rl.core.trainer import Trainer, TrainerConfig


def make_dataset(start: int, n: int) -> list[dict]:
    """Each entry pins one deterministic WebShop goal."""
    return [{"prompt": "", "answer": i} for i in range(start, start + n)]


def main() -> None:
    config_path = sys.argv[1] if len(sys.argv) > 1 else "configs/webshop.yaml"
    config = TrainerConfig.from_yaml(config_path)
    n_train = int(sys.argv[2]) if len(sys.argv) > 2 else 200
    n_val = int(sys.argv[3]) if len(sys.argv) > 3 else 64

    # verl-agent split: val [0, 500), train [500, ...]
    train = make_dataset(start=500, n=n_train)
    val = make_dataset(start=0, n=n_val)
    print(
        f"WebShop: {len(train)} train + {len(val)} val episodes "
        f"(server: {config.environment.base_url})",
        flush=True,
    )
    trainer = Trainer(config=config, dataset=train, val_dataset=val)
    trainer.train()


if __name__ == "__main__":
    main()
