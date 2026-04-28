"""MLC LLM backend for fast batched rollout generation."""

from __future__ import annotations


class MLCBackend:
    """Generation backend using MLC LLM's AsyncMLCEngine.

    MLC LLM supports continuous batching which enables significantly faster
    parallel generation compared to sequential MLX-LM generation.

    Parameters
    ----------
    model_id:
        An MLC model identifier, e.g.
        ``'HF://mlc-ai/Qwen2.5-0.5B-Instruct-q4f16_1-MLC'``.
    """

    def __init__(self, model_id: str) -> None:
        from mlc_llm import AsyncMLCEngine  # type: ignore

        self.model_id = model_id
        self.engine = AsyncMLCEngine(model_id)

    async def generate_batch(
        self, prompts: list[str], max_tokens: int = 256
    ) -> list[str]:
        """Generate responses for multiple prompts concurrently.

        Uses ``asyncio.gather`` so all prompts are submitted to the engine's
        continuous-batching scheduler at once.

        Parameters
        ----------
        prompts:
            List of plain-text prompts to generate from.
        max_tokens:
            Maximum new tokens to generate per prompt.

        Returns
        -------
        list[str]
            Generated texts in the same order as *prompts*.
        """
        import asyncio

        tasks = [self._generate_one(p, max_tokens) for p in prompts]
        return await asyncio.gather(*tasks)

    async def _generate_one(self, prompt: str, max_tokens: int) -> str:
        """Generate a single response via the async chat-completions API."""
        resp = await self.engine.chat.completions.create(
            model=self.model_id,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=0.8,
            top_p=0.95,
        )
        return resp.choices[0].message.content

    def generate_batch_sync(
        self, prompts: list[str], max_tokens: int = 256
    ) -> list[str]:
        """Synchronous wrapper around :meth:`generate_batch`.

        Runs the async coroutine in a new event loop so callers do not need
        to manage async plumbing.

        Parameters
        ----------
        prompts:
            List of plain-text prompts.
        max_tokens:
            Maximum new tokens per prompt.

        Returns
        -------
        list[str]
            Generated texts in the same order as *prompts*.
        """
        import asyncio

        return asyncio.run(self.generate_batch(prompts, max_tokens))

    def terminate(self) -> None:
        """Shut down the MLC engine and release resources."""
        self.engine.terminate()
