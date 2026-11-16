import hashlib
from dataclasses import replace

import numpy as np
import pytest

from banschik.config import Config
from banschik.memory import Memory
from banschik.storage import Store


@pytest.fixture
def config(tmp_path):
    return replace(
        Config.load(require_credentials=False),
        data_dir=tmp_path / "data",
        backup_dir=tmp_path / "backups",
        chunk_size=100,
        overlap=20,
        cooldown=0,
        allowed_chats=frozenset({-100123}),
        prompt_file=__import__("pathlib").Path("prompts/system.txt"),
    )


def fake_embed(texts, query=False):
    vectors = []
    for text in texts:
        vector = np.zeros(32, dtype="float32")
        for word in text.lower().split():
            vector[hashlib.sha256(word.encode()).digest()[0] % 32] += 1
        vectors.append(vector)
    return np.stack(vectors)


@pytest.fixture
def storage(config):
    store = Store(config.data_dir / "memory.sqlite3")
    yield store
    store.close()


@pytest.fixture
def memory(storage, config):
    return Memory(storage, config, fake_embed)


def message(text="баня завтра", chat_id=-100123, message_id=1):
    return {
        "message_id": message_id,
        "date": 1700000000,
        "chat": {"id": chat_id, "type": "supergroup"},
        "from": {"id": 42, "first_name": "Олег"},
        "text": text,
    }


def save(store, text="баня завтра", chat_id=-100123, message_id=1, update_id=1, question=None):
    store.ingest({"update_id": update_id}, message(text, chat_id, message_id), question)
