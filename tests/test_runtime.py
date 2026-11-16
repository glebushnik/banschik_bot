import asyncio
import threading

import httpx
import pytest
from apscheduler.triggers.cron import CronTrigger

from banschik.app import reply_worker
from banschik.async_utils import blocking
from banschik.llm import Assistant
from banschik.search import WebSearch
from banschik.storage import Store
from banschik.telegram import Telegram
from tests.conftest import save


async def test_reply_worker_processes_durable_mention_job(storage, memory, config):
    sent = asyncio.Event()
    save(storage, "@banschik_bot когда баня?", question="когда баня?")

    def respond(request):
        if request.url.path.endswith("/chat/completions"):
            return httpx.Response(200, json={"choices": [{"message": {"content": "В субботу."}}]})
        assert request.url.path.endswith("/sendMessage")
        sent.set()
        return httpx.Response(
            200,
            json={
                "ok": True,
                "result": {
                    "message_id": 2,
                    "date": 1700000001,
                    "chat": {"id": -100123, "type": "supergroup"},
                    "from": {"id": 99, "is_bot": True, "first_name": "Банщик"},
                    "text": "В субботу.",
                },
            },
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
        worker = asyncio.create_task(
            reply_worker(
                Assistant(client, memory, WebSearch(client, config), config),
                Telegram(client, config),
                storage,
            )
        )
        try:
            await asyncio.wait_for(sent.wait(), timeout=2)
            for _ in range(100):
                with storage.lock:
                    status = storage.db.execute("SELECT status FROM reply_jobs").fetchone()[0]
                if status == "done":
                    break
                await asyncio.sleep(0.01)
            assert status == "done"
            assert storage.stats()["messages"] == 2
            assert storage.offset() == 2
        finally:
            worker.cancel()
            with pytest.raises(asyncio.CancelledError):
                await worker


async def test_cancellation_waits_for_native_work():
    started, finish = threading.Event(), threading.Event()

    def work():
        started.set()
        finish.wait(timeout=2)

    task = asyncio.create_task(blocking(work))
    for _ in range(100):
        if started.is_set():
            break
        await asyncio.sleep(0.01)
    task.cancel()
    await asyncio.sleep(0.01)
    assert not task.done()
    finish.set()
    with pytest.raises(asyncio.CancelledError):
        await task


def test_running_job_recovers_on_restart(storage, config):
    save(storage, question="вопрос")
    job = storage.claim_job()
    storage.close()
    restarted = Store(config.data_dir / "memory.sqlite3")
    try:
        assert restarted.claim_job()["id"] == job["id"]
    finally:
        restarted.close()
        # Fixture still owns the Store object.
        storage.__init__(config.data_dir / "memory.sqlite3")


def test_nightly_cron_uses_configured_timezone(config):
    from datetime import datetime

    trigger = CronTrigger(
        hour=config.backup_hour, minute=config.backup_minute, timezone=config.timezone
    )
    now = datetime(2026, 11, 16, 1, tzinfo=config.timezone)
    next_run = trigger.get_next_fire_time(None, now)
    assert next_run.hour == 3 and next_run.minute == 0
    assert str(next_run.tzinfo) == "Europe/Moscow"
