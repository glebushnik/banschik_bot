import hashlib
import json
import os
import threading
from collections.abc import Callable

import faiss
import numpy as np

from banschik.chunks import split_text
from banschik.config import Config
from banschik.storage import Store


class LocalEmbedder:
    def __init__(self, config: Config):
        from fastembed import TextEmbedding
        from fastembed.common.model_description import ModelSource, PoolingType

        if config.embedding_model == "intfloat/multilingual-e5-small":
            supported = {item["model"] for item in TextEmbedding.list_supported_models()}
            if config.embedding_model not in supported:
                TextEmbedding.add_custom_model(
                    model=config.embedding_model,
                    pooling=PoolingType.MEAN,
                    normalization=True,
                    sources=ModelSource(hf="Xenova/multilingual-e5-small"),
                    dim=384,
                    model_file="onnx/model_quantized.onnx",
                    size_in_gb=0.118,
                    license="mit",
                )

        self.model = TextEmbedding(
            model_name=config.embedding_model,
            cache_dir=str(config.cache_dir),
            threads=config.embedding_threads,
        )
        self.e5 = "e5" in config.embedding_model.lower()
        from tokenizers import Tokenizer

        self.tokenizer = Tokenizer.from_str(self.model.model.tokenizer.to_str())
        self.tokenizer.no_truncation()
        self.tokenizer.no_padding()

    def chunks(self, text: str, size: int, overlap: int) -> list[str]:
        result = []
        for piece in split_text(text, size, overlap):
            # Leave room for E5 prefixes and special tokens. Never silently truncate a chunk.
            while len((tokens := self.tokenizer.encode(piece)).ids) > 480:
                end = tokens.offsets[470][1]
                if end <= 0:
                    raise ValueError("Tokenizer returned an invalid boundary")
                result.append(piece[:end])
                piece = piece[max(1, end - min(overlap, end // 4)) :]
            if piece.strip():
                result.append(piece.strip())
        return result

    def __call__(self, texts: list[str], query: bool = False) -> np.ndarray:
        if self.e5:
            prefix = "query: " if query else "passage: "
            texts = [prefix + text for text in texts]
        vectors = np.asarray(list(self.model.embed(texts)), dtype="float32")
        faiss.normalize_L2(vectors)
        return vectors


class Memory:
    def __init__(self, store: Store, config: Config, embedder: Callable | None = None):
        self.store = store
        self.config = config
        self.embedder = embedder
        self.inference_lock = threading.Lock()
        self.index_dir = config.data_dir / "indexes"
        self.index_dir.mkdir(parents=True, exist_ok=True)
        faiss.omp_set_num_threads(config.embedding_threads)
        self.indexes: dict[int, faiss.Index] = {}
        self.signature = json.dumps(
            {
                "model": config.embedding_model,
                "chunk_size": config.chunk_size,
                "overlap": config.overlap,
                "format": 1,
            },
            sort_keys=True,
        )
        with store.lock, store.db:
            old = store.db.execute("SELECT value FROM state WHERE key='embedding_signature'")
            row = old.fetchone()
            if row and row[0] != self.signature:
                raise ValueError(
                    "Embedding/chunk settings changed. Stop bot and run banschik reindex"
                )
            store.db.execute(
                "INSERT OR REPLACE INTO state VALUES('embedding_signature',?)", (self.signature,)
            )
        self.rebuild()

    def embed(self, texts: list[str], query: bool = False) -> np.ndarray:
        with self.inference_lock:
            if self.embedder is None:
                self.embedder = LocalEmbedder(self.config)
            vectors = np.ascontiguousarray(self.embedder(texts, query=query), dtype="float32")
            if vectors.ndim != 2 or len(vectors) != len(texts) or not np.isfinite(vectors).all():
                raise ValueError("Invalid embedding output")
            faiss.normalize_L2(vectors)
            return vectors

    def _index(self, chat_id: int, dimension: int):
        index = self.indexes.get(chat_id)
        if index is None:
            index = faiss.IndexIDMap2(faiss.IndexFlatIP(dimension))
            self.indexes[chat_id] = index
        if index.d != dimension:
            raise ValueError("Embedding dimension changed; run banschik reindex")
        return index

    def rebuild(self):
        """SQLite is authoritative; no model call needed after a crash or restore."""
        with self.store.lock:
            self.indexes.clear()
            cursor = self.store.db.execute("SELECT id,chat_id,embedding,dimension FROM chunks")
            while rows := cursor.fetchmany(512):
                groups: dict[int, list] = {}
                for row in rows:
                    groups.setdefault(row["chat_id"], []).append(row)
                for chat_id, items in groups.items():
                    dimension = items[0]["dimension"]
                    vectors = np.stack(
                        [np.frombuffer(r["embedding"], dtype="float32") for r in items]
                    )
                    self._index(chat_id, dimension).add_with_ids(
                        vectors, np.asarray([r["id"] for r in items], dtype="int64")
                    )

    def index_pending(self) -> int:
        pending = self.store.pending_messages(self.config.batch_size)
        if not pending:
            return 0
        work = []
        texts = []
        for row in pending:
            parts = split_text(row["text"], self.config.chunk_size, self.config.overlap)
            if parts:
                with self.inference_lock:
                    if self.embedder is None:
                        self.embedder = LocalEmbedder(self.config)
                    if isinstance(self.embedder, LocalEmbedder):
                        parts = self.embedder.chunks(
                            row["text"], self.config.chunk_size, self.config.overlap
                        )
            work.append((row, parts, len(texts)))
            texts.extend(parts)
        vectors = self.embed(texts) if texts else None
        try:
            with self.store.lock:
                for row, parts, start in work:
                    current = self.store.db.execute(
                        "SELECT content_hash FROM messages WHERE id=?", (row["id"],)
                    ).fetchone()
                    if not current or current[0] != row["content_hash"]:
                        continue  # An edit arrived while the embedding model was running.
                    old_ids = [
                        r[0]
                        for r in self.store.db.execute(
                            "SELECT id FROM chunks WHERE message_row_id=?", (row["id"],)
                        )
                    ]
                    index = self.indexes.get(row["chat_id"])
                    if index is not None and old_ids:
                        index.remove_ids(np.asarray(old_ids, dtype="int64"))
                    new_ids = []
                    with self.store.db:
                        self.store.db.execute(
                            "DELETE FROM chunks WHERE message_row_id=?", (row["id"],)
                        )
                        for ordinal, text in enumerate(parts):
                            vector = vectors[start + ordinal]
                            cursor = self.store.db.execute(
                                """INSERT INTO chunks
                              (message_row_id,chat_id,ordinal,text,embedding,dimension)
                              VALUES(?,?,?,?,?,?)""",
                                (
                                    row["id"],
                                    row["chat_id"],
                                    ordinal,
                                    text,
                                    vector.tobytes(),
                                    len(vector),
                                ),
                            )
                            new_ids.append(cursor.lastrowid)
                        self.store.db.execute(
                            "UPDATE messages SET indexed_hash=? WHERE id=?",
                            (row["content_hash"], row["id"]),
                        )
                    if parts:
                        self._index(row["chat_id"], vectors.shape[1]).add_with_ids(
                            vectors[start : start + len(parts)], np.asarray(new_ids, dtype="int64")
                        )
        except Exception:
            self.rebuild()
            raise
        return len(pending)

    def search(self, chat_id: int, query: str, limit: int | None = None) -> list[dict]:
        with self.store.lock:
            if chat_id not in self.indexes:
                return []
        vector = self.embed([query[:2000]], query=True)
        with self.store.lock:
            index = self.indexes[chat_id]
            _, ids = index.search(vector, limit or self.config.top_k)
            result = []
            for chunk_id in ids[0]:
                if chunk_id < 0:
                    continue
                row = self.store.db.execute(
                    """SELECT c.text AS excerpt,m.* FROM chunks c
                    JOIN messages m ON m.id=c.message_row_id
                    WHERE c.id=? AND c.chat_id=? AND m.indexed_hash=m.content_hash""",
                    (int(chunk_id), chat_id),
                ).fetchone()
                if row:
                    item = dict(row)
                    item["text"] = item.pop("excerpt")
                    item.pop("payload", None)
                    result.append(item)
            return result

    def flush(self):
        with self.store.lock:
            for chat_id, index in self.indexes.items():
                path = self.index_dir / f"{chat_id}.faiss"
                temporary = path.with_suffix(".tmp")
                faiss.write_index(index, str(temporary))
                with temporary.open("rb") as stream:
                    os.fsync(stream.fileno())
                temporary.replace(path)

    def fingerprint(self) -> str:
        return hashlib.sha256(self.signature.encode()).hexdigest()
