"""Dynamic micro-batching for the ONNX embedding/rerank workers (design: batch 32, max_wait 8 ms).

Requests arriving within `max_wait_ms` of the first queued item are executed as ONE batch call (up to `max_batch` items), so GPU/CPU
sessions see large batches under load while a lone request waits at most max_wait_ms. Each caller gets its own result or exception;
one bad item fails only that item's future when `fn` raises per-item errors, and a whole-batch failure fails every future in it
(never hangs a caller). Emits one OTel span per batch.
"""
from __future__ import annotations

import asyncio
import time
from collections.abc import Callable, Sequence
from typing import Any, Generic, TypeVar

from opentelemetry import trace

T, R = TypeVar("T"), TypeVar("R")
_tracer = trace.get_tracer("unreliable_assistant.microbatch")


class MicroBatcher(Generic[T, R]):
    def __init__(self, fn: Callable[[Sequence[T]], Sequence[R]], max_batch: int = 32, max_wait_ms: float = 8.0,
                 name: str = "batch", max_queue: int = 1024) -> None:
        if max_batch < 1 or max_wait_ms < 0:
            raise ValueError("max_batch >= 1 and max_wait_ms >= 0 required")
        self.fn, self.max_batch, self.max_wait, self.name = fn, max_batch, max_wait_ms / 1000.0, name
        self._q: asyncio.Queue[tuple[T, asyncio.Future[R]]] = asyncio.Queue(max_queue)
        self._task: asyncio.Task | None = None
        self.batch_sizes: list[int] = []

    def start(self) -> None:
        if self._task is None:
            self._task = asyncio.get_running_loop().create_task(self._run())

    async def close(self) -> None:
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None
        while not self._q.empty():  # fail anything still queued rather than leave callers hanging
            _, fut = self._q.get_nowait()
            if not fut.done():
                fut.set_exception(RuntimeError("batcher closed"))

    async def submit(self, item: T) -> R:
        self.start()
        fut: asyncio.Future[R] = asyncio.get_running_loop().create_future()
        self._q.put_nowait((item, fut))  # QueueFull propagates: callers see backpressure instead of unbounded latency
        return await fut

    async def _run(self) -> None:
        loop = asyncio.get_running_loop()
        while True:
            first = await self._q.get()
            batch = [first]
            deadline = time.monotonic() + self.max_wait
            while len(batch) < self.max_batch:
                left = deadline - time.monotonic()
                if left <= 0:
                    break
                try:
                    batch.append(await asyncio.wait_for(self._q.get(), left))
                except asyncio.TimeoutError:
                    break
            items = [i for i, _ in batch]
            self.batch_sizes.append(len(items))
            with _tracer.start_as_current_span(f"microbatch.{self.name}") as sp:
                sp.set_attribute("batch.size", len(items))
                try:
                    out: Any = await loop.run_in_executor(None, self.fn, items)
                    if len(out) != len(items):
                        raise RuntimeError("batch function returned the wrong number of results")
                except Exception as e:  # noqa: BLE001 - propagate to every waiter; never kill the loop
                    sp.set_attribute("batch.error", type(e).__name__)
                    for _, f in batch:
                        if not f.done():
                            f.set_exception(e)
                    continue
            for (_, f), r in zip(batch, out):
                if not f.done():
                    f.set_result(r)
