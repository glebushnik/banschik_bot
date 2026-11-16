import asyncio
import logging
import time
from datetime import datetime

import httpx
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
from filelock import FileLock

from banschik.async_utils import blocking
from banschik.backup import Backups
from banschik.config import Config
from banschik.llm import Assistant
from banschik.memory import Memory
from banschik.search import WebSearch
from banschik.storage import Store
from banschik.telegram import Telegram

log = logging.getLogger(__name__)


async def index_worker(memory: Memory):
    last_flush = time.monotonic()
    while True:
        try:
            count = await blocking(memory.index_pending)
            if time.monotonic() - last_flush > 30:
                await blocking(memory.flush)
                last_flush = time.monotonic()
            await asyncio.sleep(0.05 if count else memory.config.index_interval)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            log.error("Indexing failed (%s); retrying in 10 seconds", type(exc).__name__)
            await asyncio.sleep(10)


async def reply_worker(assistant: Assistant, telegram: Telegram, store: Store):
    while True:
        job = await blocking(store.claim_job)
        if not job:
            await asyncio.sleep(0.25)
            continue
        try:
            answer = await assistant.answer(job["chat_id"], job["question"])
            async for sent in telegram.send(
                job["chat_id"], job["message_id"], answer, job["topic_id"]
            ):
                await blocking(store.ingest, {}, sent, None, advance_offset=False)
            await blocking(store.finish_job, job["id"])
        except asyncio.CancelledError:
            # Leave running job for restart recovery.
            raise
        except Exception as exc:
            log.error("Reply job %s failed (%s)", job["id"], type(exc).__name__)
            if job["attempts"] >= 2:
                await blocking(store.finish_job, job["id"], "failed")
                try:
                    async for _ in telegram.send(
                        job["chat_id"],
                        job["message_id"],
                        "Сейчас не удалось ответить. Попробуй позже.",
                        job["topic_id"],
                    ):
                        pass
                except Exception:
                    log.error("Could not send failure notice for job %s", job["id"])
            else:
                await blocking(store.finish_job, job["id"], "pending", 15)


async def run(config: Config):
    config.data_dir.mkdir(parents=True, exist_ok=True)
    with FileLock(str(config.data_dir / "runtime.lock"), timeout=0):
        store = Store(config.data_dir / "memory.sqlite3")
        scheduler = BackgroundScheduler(timezone=config.timezone)
        tasks = []
        try:
            memory = await blocking(Memory, store, config)
            backups = Backups(store, memory, config)
            scheduler.add_job(
                backups.create,
                CronTrigger(
                    hour=config.backup_hour, minute=config.backup_minute, timezone=config.timezone
                ),
                id="nightly-backup",
                coalesce=True,
                max_instances=1,
                misfire_grace_time=21600,
            )
            scheduler.start()
            # Catch up if the process was down at the nightly backup time.
            now = datetime.now(config.timezone)
            target = config.backup_dir / f"banschik-{now.date().isoformat()}.tar.gz"
            if (now.hour, now.minute) >= (
                config.backup_hour,
                config.backup_minute,
            ) and not target.exists():
                await blocking(backups.create)
            async with httpx.AsyncClient(follow_redirects=False) as client:
                telegram = Telegram(client, config)
                me = await telegram.call("getMe")
                info = await telegram.call("getWebhookInfo")
                if info.get("url"):
                    raise ValueError("Bot has a webhook. Remove it before running the polling bot")
                assistant = Assistant(client, memory, WebSearch(client, config), config)
                log.info(
                    "Bot @%s started; allowed groups: %s",
                    me["username"],
                    sorted(config.allowed_chats),
                )
                tasks = [asyncio.create_task(index_worker(memory))]
                tasks.extend(
                    asyncio.create_task(reply_worker(assistant, telegram, store))
                    for _ in range(config.concurrency)
                )
                tasks.append(asyncio.create_task(telegram.poll(store, me["username"])))
                await asyncio.gather(*tasks)
        finally:
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            if scheduler.running:
                await blocking(scheduler.shutdown, wait=True)
            # Wait for any in-flight embedding work before closing the database.
            if "memory" in locals():
                await blocking(_finish_memory, memory)
            store.close()


def _finish_memory(memory: Memory):
    with memory.inference_lock:
        memory.flush()
