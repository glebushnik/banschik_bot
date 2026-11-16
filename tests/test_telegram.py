import asyncio
import json

import httpx
import pytest

from banschik.telegram import Telegram, mentioned_question
from tests.conftest import message


def with_mention(text, username="banschik_bot"):
    msg = message(text)
    offset = len(text[: text.index("@")].encode("utf-16-le")) // 2
    msg["entities"] = [{"type": "mention", "offset": offset, "length": len(username) + 1}]
    return msg


def test_only_explicit_mention_triggers():
    assert mentioned_question(message("Привет"), "banschik_bot") is None
    reply = message("а когда?")
    reply["reply_to_message"] = {"from": {"is_bot": True}}
    assert mentioned_question(reply, "banschik_bot") is None
    assert (
        mentioned_question(with_mention("@other_bot привет", "other_bot"), "banschik_bot") is None
    )
    assert mentioned_question(with_mention("@Banschik_bot привет"), "banschik_bot") == "привет"
    assert (
        mentioned_question(with_mention("😀 @banschik_bot когда?"), "banschik_bot") == "😀  когда?"
    )


def test_caption_mention():
    msg = with_mention("@banschik_bot вопрос")
    msg["caption"] = msg.pop("text")
    msg["caption_entities"] = msg.pop("entities")
    assert mentioned_question(msg, "banschik_bot") == "вопрос"


async def test_poll_records_silently_and_queues_only_mentions(storage, config):
    plain = message("обычное обсуждение")
    mention = with_mention("@banschik_bot вопрос")
    mention["message_id"] = 2
    other_group = with_mention("@banschik_bot секрет")
    other_group["chat"]["id"] = -999
    edited = with_mention("@banschik_bot правка")
    edited["message_id"] = 3
    private = with_mention("@banschik_bot личное")
    private["chat"]["type"] = "private"
    calls = []

    def respond(request):
        calls.append(request.url.path.rsplit("/", 1)[-1])
        if len(calls) > 1:
            raise asyncio.CancelledError
        return httpx.Response(
            200,
            json={
                "ok": True,
                "result": [
                    {"update_id": 1, "message": plain},
                    {"update_id": 2, "message": mention},
                    {"update_id": 3, "message": other_group},
                    {"update_id": 4, "edited_message": edited},
                    {"update_id": 5, "message": private},
                ],
            },
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
        telegram = Telegram(client, config)
        with pytest.raises(asyncio.CancelledError):
            await telegram.poll(storage, "banschik_bot")
    assert storage.stats()["messages"] == 3
    assert storage.stats()["reply_jobs"] == 1
    assert storage.offset() == 6
    assert set(calls) == {"getUpdates"}  # No outgoing messages while ingesting.


async def test_reply_splitting_respects_utf16_and_thread(config):
    bodies = []

    def respond(request):
        bodies.append(json.loads(request.content))
        return httpx.Response(200, json={"ok": True, "result": {"message_id": 9}})

    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
        telegram = Telegram(client, config)
        async for _ in telegram.send(-100123, 8, "😀" * 3000, topic_id=4):
            pass
    assert len(bodies) == 2
    assert all(len(body["text"].encode("utf-16-le")) // 2 <= 4096 for body in bodies)
    assert all(body["message_thread_id"] == 4 for body in bodies)
