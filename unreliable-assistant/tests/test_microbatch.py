import asyncio
import time

import pytest

from app.microbatch import MicroBatcher


async def test_concurrent_requests_are_coalesced_up_to_max_batch():
    calls = []

    def fn(items):
        calls.append(len(items))
        return [i * 2 for i in items]

    mb = MicroBatcher(fn, max_batch=32, max_wait_ms=8)
    res = await asyncio.gather(*[mb.submit(i) for i in range(70)])
    await mb.close()
    assert res == [i * 2 for i in range(70)]            # each caller gets its own result, in order
    assert max(calls) == 32 and sum(calls) == 70 and len(calls) <= 4


async def test_lone_request_waits_at_most_max_wait():
    mb = MicroBatcher(lambda items: list(items), max_batch=32, max_wait_ms=8)
    t = time.perf_counter()
    assert await mb.submit("x") == "x"
    ms = (time.perf_counter() - t) * 1000
    await mb.close()
    assert 6 <= ms < 60, ms                              # waited for the window, not forever


async def test_batch_failure_reaches_every_waiter_and_loop_survives():
    state = {"n": 0}

    def fn(items):
        state["n"] += 1
        if state["n"] == 1:
            raise ValueError("boom")
        return list(items)

    mb = MicroBatcher(fn, max_batch=8, max_wait_ms=5)
    bad = await asyncio.gather(*[mb.submit(i) for i in range(3)], return_exceptions=True)
    assert all(isinstance(b, ValueError) for b in bad)
    assert await mb.submit(7) == 7                       # next batch still works
    await mb.close()


async def test_wrong_result_count_is_an_error_not_a_hang():
    mb = MicroBatcher(lambda items: [1], max_batch=8, max_wait_ms=5)
    r = await asyncio.gather(mb.submit(1), mb.submit(2), return_exceptions=True)
    await mb.close()
    assert all(isinstance(x, RuntimeError) for x in r)


async def test_full_queue_applies_backpressure():
    mb = MicroBatcher(lambda items: list(items), max_batch=1, max_wait_ms=0, max_queue=2)
    mb.start()
    mb._task.cancel()  # stop draining so the queue fills deterministically
    held = [asyncio.ensure_future(mb.submit(i)) for i in range(2)]
    await asyncio.sleep(0)
    with pytest.raises(asyncio.QueueFull):
        await mb.submit(99)
    for h in held:
        h.cancel()
