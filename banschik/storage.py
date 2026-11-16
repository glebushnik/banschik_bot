import hashlib
import json
import sqlite3
import threading
import time
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS state (key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS messages (
 id INTEGER PRIMARY KEY, chat_id INTEGER NOT NULL, message_id INTEGER NOT NULL,
 topic_id INTEGER, sender_id INTEGER, author TEXT NOT NULL, sent_at INTEGER NOT NULL,
 text TEXT NOT NULL, payload TEXT NOT NULL, content_hash TEXT NOT NULL,
 indexed_hash TEXT, UNIQUE(chat_id, message_id)
);
CREATE INDEX IF NOT EXISTS messages_chat_time ON messages(chat_id, sent_at, message_id);
CREATE TABLE IF NOT EXISTS chunks (
 id INTEGER PRIMARY KEY AUTOINCREMENT,
 message_row_id INTEGER NOT NULL REFERENCES messages(id) ON DELETE CASCADE,
 chat_id INTEGER NOT NULL, ordinal INTEGER NOT NULL, text TEXT NOT NULL,
 embedding BLOB NOT NULL, dimension INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS chunks_chat ON chunks(chat_id);
CREATE TABLE IF NOT EXISTS reply_jobs (
 id INTEGER PRIMARY KEY, update_id INTEGER UNIQUE NOT NULL,
 chat_id INTEGER NOT NULL, message_id INTEGER NOT NULL, topic_id INTEGER,
 question TEXT NOT NULL, status TEXT NOT NULL DEFAULT 'pending',
 attempts INTEGER NOT NULL DEFAULT 0, created_at INTEGER NOT NULL,
 next_attempt INTEGER NOT NULL DEFAULT 0
);
"""


class Store:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self.lock = threading.RLock()
        self.db = sqlite3.connect(path, check_same_thread=False, timeout=30)
        self.db.row_factory = sqlite3.Row
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("PRAGMA synchronous=FULL")
        self.db.execute("PRAGMA foreign_keys=ON")
        self.db.executescript(SCHEMA)
        with self.db:
            self.db.execute("UPDATE reply_jobs SET status='pending' WHERE status='running'")

    def close(self):
        with self.lock:
            self.db.close()

    def offset(self) -> int:
        with self.lock:
            row = self.db.execute("SELECT value FROM state WHERE key='offset'").fetchone()
            return int(row[0]) if row else 0

    def ingest(
        self,
        update: dict,
        message: dict | None,
        question: str | None,
        cooldown: int = 0,
        advance_offset: bool = True,
    ):
        """Persist message, reply job and polling offset in one transaction."""
        with self.lock, self.db:
            if message is not None:
                text = message.get("text") or message.get("caption") or ""
                sender = message.get("from") or message.get("sender_chat") or {}
                author = (
                    sender.get("username")
                    or " ".join(
                        filter(
                            None,
                            [
                                sender.get("first_name"),
                                sender.get("last_name"),
                                sender.get("title"),
                            ],
                        )
                    )
                    or "unknown"
                )
                digest = hashlib.sha256(text.encode()).hexdigest()
                self.db.execute(
                    """
                  INSERT INTO messages(chat_id,message_id,topic_id,sender_id,author,sent_at,
                    text,payload,content_hash) VALUES(?,?,?,?,?,?,?,?,?)
                  ON CONFLICT(chat_id,message_id) DO UPDATE SET
                    topic_id=excluded.topic_id, author=excluded.author, text=excluded.text,
                    payload=excluded.payload, content_hash=excluded.content_hash
                """,
                    (
                        message["chat"]["id"],
                        message["message_id"],
                        message.get("message_thread_id"),
                        sender.get("id"),
                        author,
                        message["date"],
                        text,
                        json.dumps(message, ensure_ascii=False),
                        digest,
                    ),
                )
                if question is not None:
                    chat_id = message["chat"]["id"]
                    last = self.db.execute(
                        "SELECT MAX(created_at) FROM reply_jobs WHERE chat_id=?", (chat_id,)
                    ).fetchone()[0]
                    if last is None or time.time() - last >= cooldown:
                        self.db.execute(
                            """
                          INSERT OR IGNORE INTO reply_jobs(update_id,chat_id,message_id,
                            topic_id,question,created_at) VALUES(?,?,?,?,?,?)
                        """,
                            (
                                update["update_id"],
                                chat_id,
                                message["message_id"],
                                message.get("message_thread_id"),
                                question,
                                int(time.time()),
                            ),
                        )
            if advance_offset:
                self.db.execute(
                    "INSERT OR REPLACE INTO state(key,value) VALUES('offset',?)",
                    (str(update["update_id"] + 1),),
                )

    def pending_messages(self, limit: int):
        with self.lock:
            return self.db.execute(
                """SELECT * FROM messages
                WHERE indexed_hash IS NULL OR indexed_hash != content_hash
                ORDER BY id LIMIT ?""",
                (limit,),
            ).fetchall()

    def claim_job(self):
        with self.lock, self.db:
            self.db.execute(
                "UPDATE reply_jobs SET status='expired' WHERE status='pending' AND created_at < ?",
                (int(time.time()) - 21600,),
            )
            # Keep answers inside each group in order, while other groups run concurrently.
            row = self.db.execute(
                """SELECT * FROM reply_jobs j
              WHERE status='pending' AND next_attempt <= ?
              AND NOT EXISTS (SELECT 1 FROM reply_jobs r
                WHERE r.chat_id=j.chat_id AND r.status='running')
              ORDER BY id LIMIT 1""",
                (int(time.time()),),
            ).fetchone()
            if row:
                self.db.execute(
                    "UPDATE reply_jobs SET status='running',attempts=attempts+1 WHERE id=?",
                    (row["id"],),
                )
                return dict(row)
            return None

    def finish_job(self, job_id: int, status: str = "done", delay: int = 0):
        with self.lock, self.db:
            self.db.execute(
                "UPDATE reply_jobs SET status=?,next_attempt=? WHERE id=?",
                (status, int(time.time()) + delay, job_id),
            )

    def recent(self, chat_id: int, limit: int):
        with self.lock:
            return self.db.execute(
                """SELECT * FROM (SELECT * FROM messages
                WHERE chat_id=? AND text != '' ORDER BY sent_at DESC,message_id DESC LIMIT ?)
                ORDER BY sent_at,message_id""",
                (chat_id, limit),
            ).fetchall()

    def snapshot(self, destination: Path):
        with self.lock, sqlite3.connect(destination) as target:
            self.db.backup(target)

    def stats(self):
        with self.lock:
            return {
                name: self.db.execute(f"SELECT count(*) FROM {name}").fetchone()[0]
                for name in ("messages", "chunks", "reply_jobs")
            }
