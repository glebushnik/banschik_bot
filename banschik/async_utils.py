import asyncio


async def blocking(function, *args, **kwargs):
    """Let a running native/SQLite call finish before cancellation closes its resources."""
    task = asyncio.create_task(asyncio.to_thread(function, *args, **kwargs))
    try:
        return await asyncio.shield(task)
    except asyncio.CancelledError:
        await task
        raise
