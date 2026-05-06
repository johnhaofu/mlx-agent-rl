"""Train a model to use calculator tool on GSM8K."""

import os
import re
import sys

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

from datasets import load_dataset  # noqa: E402

from mlx_agent_rl.core.trainer import Trainer, TrainerConfig  # noqa: E402


def load_gsm8k(split: str = "train", max_samples: int = 500) -> list[dict]:
    """Load GSM8K and return list of {prompt, answer} dicts."""
    ds = load_dataset("openai/gsm8k", "main", split=split)
    data = []
    for item in ds:
        match = re.search(r"####\s*(-?\d[\d,]*)", item["answer"])
        if match:
            data.append(
                {
                    "prompt": item["question"],
                    "answer": match.group(1).replace(",", ""),
                }
            )
    return data[:max_samples]


def main() -> None:
    config_path = sys.argv[1] if len(sys.argv) > 1 else "configs/gsm8k_calculator.yaml"
    config = TrainerConfig.from_yaml(config_path)
    max_samples = int(sys.argv[2]) if len(sys.argv) > 2 else 20
    val_size = int(sys.argv[3]) if len(sys.argv) > 3 else 64

    dataset = load_gsm8k(split="train", max_samples=max_samples)
    val_dataset = load_gsm8k(split="test", max_samples=val_size)
    print(
        f"Loaded {len(dataset)} train + {len(val_dataset)} val examples",
        flush=True,
    )
    trainer = Trainer(config=config, dataset=dataset, val_dataset=val_dataset)
    trainer.train()


if __name__ == "__main__":
    main()
