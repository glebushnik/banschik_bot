import asyncio
import logging

import httpx

from banschik.async_utils import blocking
from banschik.config import Config
from banschik.storage import Store

log = logging.getLogger(__name__)


def mentioned_question(message: dict, username: str) -> str | None:
    """Use Telegram entities and UTF-16 offsets; replies alone never trigger an answer."""
    text = message.get("text") or message.get("caption") or ""
    entities = message.get("entities") or message.get("caption_entities") or []
    encoded = text.encode("utf-16-le")
    spans = []
    for entity in entities:
        if entity.get("type") != "mention":
            continue
        start = entity["offset"] * 2
        end = start + entity["length"] * 2
        if encoded[start:end].decode("utf-16-le").casefold() == f"@{username}".casefold():
            spans.append((start, end))
    if not spans:
        return None
    for start, end in sorted(spans, reverse=True):
        encoded = encoded[:start] + encoded[end:]
    return encoded.decode("utf-16-le").strip() or "Привет!"


class TelegramError(Exception):
    pass


class Telegram:
    def __init__(self, client: httpx.AsyncClient, config: Config):
        self.client, self.config = client, config
        self.base = f"https://api.telegram.org/bot{config.token}"

    async def call(self, method: str, payload: dict | None = None):
        for _attempt in range(3):
            response = await self.client.post(
                f"{self.base}/{method}", json=payload or {}, timeout=45
            )
            if response.status_code == 429:
                delay = min(response.json().get("parameters", {}).get("retry_after", 5), 60)
                await asyncio.sleep(delay)
                continue
            # Never include the URL in errors: it contains the bot token.
            if response.status_code >= 400:
                raise TelegramError(f"Telegram {method}: HTTP {response.status_code}")
            data = response.json()
            if not data.get("ok"):
                raise TelegramError(f"Telegram {method} failed")
            return data["result"]
        raise TelegramError(f"Telegram {method}: rate limit")

    async def send(self, chat_id: int, message_id: int, text: str, topic_id: int | None = None):
        # Telegram limits message text to 4096 UTF-16 code units.
        parts = []
        current, units = [], 0
        for char in text:
            length = len(char.encode("utf-16-le")) // 2
            if units + length > 3900:
                parts.append("".join(current))
                current, units = [], 0
            current.append(char)
            units += length
        if current:
            parts.append("".join(current))
        for part in parts:
            payload = {
                "chat_id": chat_id,
                "text": part,
                "reply_parameters": {"message_id": message_id, "allow_sending_without_reply": True},
                "link_preview_options": {"is_disabled": True},
            }
            if topic_id is not None:
                payload["message_thread_id"] = topic_id
            yield await self.call("sendMessage", payload)

    async def poll(self, store: Store, username: str):
        while True:
            try:
                updates = await self.call(
                    "getUpdates",
                    {
                        "offset": store.offset(),
                        "timeout": 30,
                        "allowed_updates": ["message", "edited_message"],
                    },
                )
                for update in updates:
                    if update["update_id"] < store.offset():
                        continue
                    message = update.get("message") or update.get("edited_message")
                    question = None
                    if message:
                        chat = message["chat"]
                        if chat["type"] not in {"group", "supergroup"}:
                            message = None
                        elif chat["id"] not in self.config.allowed_chats:
                            log.warning("Ignoring group outside allowlist: %s", chat["id"])
                            message = None
                        elif "message" in update and not message.get("from", {}).get("is_bot"):
                            question = mentioned_question(message, username)
                    await blocking(store.ingest, update, message, question, self.config.cooldown)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                log.error("Polling failed (%s); retrying in 5 seconds", type(exc).__name__)
                await asyncio.sleep(5)
