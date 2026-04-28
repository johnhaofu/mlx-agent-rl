"""llama.cpp backend for fast batched rollout generation via llama-server."""

from __future__ import annotations

import asyncio
import os
import subprocess
import time

import aiohttp
import requests


class LlamaCppBackend:
    """Manages a llama-server subprocess and generates text via its OpenAI-compatible API.

    Parameters
    ----------
    gguf_path:
        Path to the GGUF model file.
    port:
        Port for the llama-server HTTP API.
    n_parallel:
        Number of parallel generation slots (continuous batching).
    n_ctx:
        Total context length across all parallel slots.
    """

    def __init__(
        self,
        gguf_path: str,
        port: int = 8090,
        n_parallel: int = 16,
        n_ctx: int = 8192,
    ) -> None:
        self.gguf_path = gguf_path
        self.port = port
        self.n_parallel = n_parallel
        self.n_ctx = n_ctx
        self.base_url = f"http://localhost:{port}"
        self._process = None
        self.start_server()

    # ------------------------------------------------------------------
    # Server lifecycle
    # ------------------------------------------------------------------

    def start_server(self) -> None:
        """Start llama-server as a subprocess."""
        self.stop_server()  # kill any existing instance
        self._process = subprocess.Popen(
            [
                "llama-server",
                "-m", self.gguf_path,
                "--port", str(self.port),
                "-np", str(self.n_parallel),
                "-c", str(self.n_ctx),
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        # Wait up to 30 seconds for the server to become ready
        for _ in range(30):
            try:
                resp = requests.get(f"{self.base_url}/health", timeout=1)
                if resp.status_code == 200:
                    return
            except Exception:
                pass
            time.sleep(1)
        raise RuntimeError("llama-server failed to start within 30 seconds")

    def stop_server(self) -> None:
        """Stop the llama-server subprocess."""
        if self._process is not None:
            self._process.terminate()
            try:
                self._process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self._process.kill()
            self._process = None
        # Also kill any orphan llama-server processes on our port
        os.system(f"lsof -ti:{self.port} | xargs kill 2>/dev/null")

    def restart_with_model(self, gguf_path: str) -> None:
        """Restart the server with a new model file (e.g. after weight sync)."""
        self.gguf_path = gguf_path
        self.start_server()

    def terminate(self) -> None:
        """Alias for stop_server."""
        self.stop_server()

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------

    def generate_batch_sync(self, prompts: list[str], max_tokens: int = 256) -> list[str]:
        """Generate responses for multiple prompts using async HTTP requests.

        llama-server handles continuous batching internally; all requests are
        dispatched concurrently via asyncio + aiohttp.
        """
        return asyncio.run(self._generate_batch(prompts, max_tokens))

    async def _generate_batch(self, prompts: list[str], max_tokens: int) -> list[str]:
        url = f"{self.base_url}/v1/chat/completions"
        async with aiohttp.ClientSession() as session:
            tasks = [
                self._generate_one(session, url, prompt, max_tokens)
                for prompt in prompts
            ]
            return await asyncio.gather(*tasks)

    async def _generate_one(
        self,
        session: aiohttp.ClientSession,
        url: str,
        prompt: str,
        max_tokens: int,
    ) -> str:
        payload = {
            "model": "local",
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": 0.8,
            "top_p": 0.95,
        }
        async with session.post(url, json=payload) as resp:
            data = await resp.json()
            return data["choices"][0]["message"]["content"]
