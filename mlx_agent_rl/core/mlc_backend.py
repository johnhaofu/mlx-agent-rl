"""MLC LLM backend for fast batched rollout generation."""

from __future__ import annotations

import asyncio
import threading


class MLCBackend:
    """Generation backend using MLC LLM's AsyncMLCEngine.

    Runs a persistent event loop in a background thread so the async engine
    stays alive across multiple generate_batch_sync() calls.

    Parameters
    ----------
    model_id:
        An MLC model identifier, e.g.
        ``'HF://mlc-ai/Qwen2.5-0.5B-Instruct-q4f16_1-MLC'``.
    """

    def __init__(self, model_id: str) -> None:
        self.model_id = model_id

        # Create a persistent event loop in a background thread
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()

        # Create engine on the background loop
        self._engine = asyncio.run_coroutine_threadsafe(
            self._create_engine(), self._loop
        ).result()

    def _run_loop(self) -> None:
        asyncio.set_event_loop(self._loop)
        self._loop.run_forever()

    async def _create_engine(self):
        from mlc_llm import AsyncMLCEngine  # type: ignore
        return AsyncMLCEngine(self.model_id)

    def generate_batch_sync(
        self, prompts: list[str], max_tokens: int = 256
    ) -> list[str]:
        """Generate responses for multiple prompts concurrently.

        Submits all prompts to the MLC engine's continuous-batching scheduler
        at once via the persistent background event loop.
        """
        future = asyncio.run_coroutine_threadsafe(
            self._generate_batch(prompts, max_tokens), self._loop
        )
        return future.result()

    async def _generate_batch(
        self, prompts: list[str], max_tokens: int
    ) -> list[str]:
        tasks = [self._generate_one(p, max_tokens) for p in prompts]
        return await asyncio.gather(*tasks)

    async def _generate_one(self, prompt: str, max_tokens: int) -> str:
        resp = await self._engine.chat.completions.create(
            model=self.model_id,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=0.8,
            top_p=0.95,
        )
        return resp.choices[0].message.content

    def terminate(self) -> None:
        """Shut down the MLC engine and stop the background loop."""
        asyncio.run_coroutine_threadsafe(
            self._engine.terminate(), self._loop
        ).result(timeout=10)
        self._loop.call_soon_threadsafe(self._loop.stop)
        self._thread.join(timeout=5)
