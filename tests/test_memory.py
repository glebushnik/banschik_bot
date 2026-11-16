from dataclasses import replace

import faiss
import pytest

from banschik.chunks import split_text
from banschik.memory import Memory
from tests.conftest import fake_embed, save


def test_chunking_preserves_unicode_and_overlap():
    text = "баня " * 100
    chunks = split_text(text, 100, 20)
    assert len(chunks) > 1
    assert all(0 < len(chunk) <= 100 for chunk in chunks)
    assert chunks[0][-19:] in chunks[1]
    with pytest.raises(ValueError):
        split_text(text, 10, 10)


def test_index_search_isolation_and_edits(storage, memory):
    save(storage, "баня завтра")
    save(storage, "секрет другой группы", chat_id=-999, update_id=2)
    assert memory.index_pending() == 2
    found = memory.search(-100123, "баня")
    assert len(found) == 1 and found[0]["text"] == "баня завтра"
    assert not memory.search(-666, "баня")
    save(storage, "встреча отменена", update_id=3)
    assert not memory.search(-100123, "баня")  # Stale chunks hidden before reindex.
    memory.index_pending()
    assert memory.indexes[-100123].ntotal == 1
    assert memory.search(-100123, "отменена")[0]["text"] == "встреча отменена"


def test_restart_rebuilds_from_sqlite_without_embedding(storage, memory, config):
    save(storage)
    memory.index_pending()
    memory.flush()
    path = config.data_dir / "indexes" / "-100123.faiss"
    assert faiss.read_index(str(path)).ntotal == 1
    path.write_bytes(b"broken cache")
    restarted = Memory(storage, config, fake_embed)
    assert restarted.search(-100123, "баня")
    restarted.flush()
    assert faiss.read_index(str(path)).ntotal == 1


def test_settings_change_requires_reindex(storage, memory, config):
    with pytest.raises(ValueError, match="reindex"):
        Memory(storage, replace(config, overlap=10), fake_embed)


def test_edit_during_embedding_is_not_marked_indexed(storage, config):
    save(storage, "старое")

    def embed(texts, query=False):
        save(storage, "новое", update_id=2)
        return fake_embed(texts)

    mem = Memory(storage, config, embed)
    mem.index_pending()
    assert storage.pending_messages(10)[0]["text"] == "новое"
    assert storage.stats()["chunks"] == 0


def test_message_offset_and_job_dedup_are_durable(storage):
    save(storage, question="когда баня?", update_id=8)
    save(storage, question="когда баня?", update_id=8)
    assert storage.offset() == 9
    assert storage.stats() == {"messages": 1, "chunks": 0, "reply_jobs": 1}
    job = storage.claim_job()
    assert job["question"] == "когда баня?"
    assert storage.claim_job() is None
    storage.finish_job(job["id"])


def test_bot_reply_does_not_move_polling_offset(storage):
    save(storage, update_id=8)
    from tests.conftest import message

    storage.ingest({}, message("ответ", message_id=2), None, advance_offset=False)
    assert storage.offset() == 9
