# mlx-agent-rl

MLX-native multi-turn agent reinforcement learning framework for Apple Silicon.

Train language models to use tools (calculator, search, etc.) through multi-turn RL on your Mac.

## Features

- **Multi-turn rollout**: model generates actions, environment responds, model continues
- **GRPO family algorithms**: GRPO, Dr.GRPO, DAPO, GiGPO (all critic-free)
- **GiGPO**: two-level advantage — episode-level + step-level anchor state grouping (NeurIPS 2025)
- **Step-independent prompts**: sliding window memory keeps context length constant
- **LoRA + quantization**: efficient training on limited Mac memory
- **Built on mlx-lm**: no CUDA dependencies, pure Apple Silicon

## Architecture

```
Trainer
  ├── RolloutCollector        # Multi-turn trajectory collection
  │     ├── Policy            # mlx-lm model with LoRA
  │     ├── Environment       # Tool environments (calculator, search)
  │     └── SlidingMemory     # Sliding window history
  ├── AdvantageEstimator      # GRPO / Dr.GRPO / DAPO / GiGPO
  └── PPO-clip update         # Policy gradient with clipping
```

## Quick Start

```bash
pip install -e ".[dev]"
python examples/train_gsm8k.py configs/gsm8k_calculator.yaml
```

## How It Works

Each training step:

1. **Rollout** — For each math problem, the model interacts with a calculator tool over multiple turns:
   ```
   Model: <think>I need to divide 48 by 2</think><action>calculate(48/2)</action>
   Env:   Result: 24
   Model: <think>Now add 48 + 24</think><action>calculate(48+24)</action>
   Env:   Result: 72
   Model: <action>answer(72)</action>
   Env:   Correct! reward=1.0
   ```

2. **Advantage** — Group trajectories by prompt, compute normalized advantages (GRPO/GiGPO)

3. **Update** — PPO-clip objective updates LoRA weights

## Configuration

```yaml
model:
  path: /path/to/model        # Any mlx-lm compatible model
  quantize: 4                  # 4/8/null
  lora:
    rank: 8
    layers: 4

rollout:
  group_size: 4                # Trajectories per prompt
  max_steps: 5                 # Max tool interactions per episode
  max_tokens: 256              # Max generation per step

training:
  algorithm: dapo              # grpo / dr_grpo / dapo / gigpo
  lr: 1e-6
  epochs: 10
  batch_size: 2
  epsilon: 1e-4
  epsilon_high: 1e-2           # DAPO dual clipping

environment:
  type: calculator
  invalid_action_penalty: -0.1

memory:
  window_size: 3               # Sliding window size
```

## Algorithms

| Algorithm | Key Idea |
|-----------|----------|
| **GRPO** | Group-normalized episode reward |
| **Dr.GRPO** | Mean-only normalization (no std), more stable |
| **DAPO** | GRPO + dual epsilon clipping for flexible policy updates |
| **GiGPO** | GRPO + step-level anchor state grouping for fine-grained credit assignment |

## Adding Custom Environments

```python
from mlx_agent_rl.environments.base import BaseEnvironment, Observation

class MyEnvironment(BaseEnvironment):
    def reset(self, prompt, **kwargs) -> Observation:
        return Observation(text="initial state", anchor="state_id")

    def step(self, action) -> tuple[Observation, float, bool]:
        # Execute action, return (observation, reward, done)
        ...

    def extract_action(self, model_output) -> str | None:
        # Parse <action>...</action> from model text
        ...
```

## Tests

```bash
pytest tests/ -v
```

## Inspired By

- [verl-agent](https://github.com/langfengQ/verl-agent) — multi-turn agent RL with GiGPO
- [mlx-lm-lora](https://github.com/Goekdeniz-Guelmez/mlx-lm-lora) — MLX fine-tuning framework
- [DeepSeek-R1](https://arxiv.org/abs/2501.12948) — GRPO for reasoning

## License

MIT
