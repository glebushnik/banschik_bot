import hashlib
import json
import os
import re
import sqlite3
import tarfile
import tempfile
import threading
from datetime import datetime, timedelta
from pathlib import Path

from banschik.config import Config
from banschik.memory import Memory
from banschik.storage import Store

BACKUP_NAME = re.compile(r"^banschik-(\d{4}-\d{2}-\d{2})\.tar\.gz$")


def checksum(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class Backups:
    def __init__(self, store: Store, memory: Memory, config: Config):
        self.store, self.memory, self.config = store, memory, config
        self.lock = threading.Lock()
        config.backup_dir.mkdir(parents=True, exist_ok=True)

    def create(self, now: datetime | None = None) -> Path:
        with self.lock:
            now = (now or datetime.now(self.config.timezone)).astimezone(self.config.timezone)
            target = self.config.backup_dir / f"banschik-{now.date().isoformat()}.tar.gz"
            with tempfile.TemporaryDirectory(dir=self.config.backup_dir) as directory:
                stage = Path(directory)
                (stage / "indexes").mkdir()
                # The same lock guards SQLite commits and index mutations.
                with self.store.lock:
                    self.store.snapshot(stage / "memory.sqlite3")
                    import faiss

                    for chat_id, index in self.memory.indexes.items():
                        faiss.write_index(index, str(stage / "indexes" / f"{chat_id}.faiss"))
                files = {
                    str(p.relative_to(stage)): checksum(p) for p in stage.rglob("*") if p.is_file()
                }
                manifest = {
                    "format": 1,
                    "created_at": now.isoformat(),
                    "embedding_signature": self.memory.signature,
                    "files": files,
                }
                (stage / "manifest.json").write_text(json.dumps(manifest, indent=2))
                archive = stage / "snapshot.tar.gz"
                with tarfile.open(archive, "w:gz") as tar:
                    for name in [*files, "manifest.json"]:
                        tar.add(stage / name, arcname=name)
                verify(archive)
                with archive.open("rb") as stream:
                    os.fsync(stream.fileno())
                archive.replace(target)
            # Never prune until a new complete, verified snapshot exists.
            cutoff = now.date() - timedelta(days=self.config.retention_days - 1)
            for path in self.config.backup_dir.iterdir():
                match = BACKUP_NAME.fullmatch(path.name)
                if match and path.is_file():
                    try:
                        day = datetime.strptime(match[1], "%Y-%m-%d").date()
                    except ValueError:
                        continue
                    if day < cutoff:
                        path.unlink()
            return target


def verify(archive: Path) -> dict:
    with tarfile.open(archive, "r:gz") as tar:
        manifest = json.load(tar.extractfile("manifest.json"))
        if manifest.get("format") != 1 or "memory.sqlite3" not in manifest.get("files", {}):
            raise ValueError("Unsupported backup format")
        for name, expected in manifest["files"].items():
            if name != "memory.sqlite3" and not re.fullmatch(r"indexes/-?\d+\.faiss", name):
                raise ValueError("Unexpected backup member")
            member = tar.getmember(name)
            if not member.isfile():
                raise ValueError("Backup member is not a regular file")
            digest = hashlib.sha256()
            with tar.extractfile(member) as stream:
                for block in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(block)
            if digest.hexdigest() != expected:
                raise ValueError(f"Backup checksum mismatch: {name}")
        return manifest


def restore(archive: Path, destination: Path):
    manifest = verify(archive)
    if destination.exists() and any(destination.iterdir()):
        raise ValueError("Restore destination must be empty; stop bot and choose a new directory")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=destination.parent) as directory:
        stage = Path(directory)
        with tarfile.open(archive, "r:gz") as tar:
            for name in manifest["files"]:
                path = stage / name
                path.parent.mkdir(parents=True, exist_ok=True)
                with tar.extractfile(name) as source, path.open("wb") as target:
                    for block in iter(lambda: source.read(1024 * 1024), b""):
                        target.write(block)
        with sqlite3.connect(stage / "memory.sqlite3") as db:
            if db.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
                raise ValueError("SQLite integrity check failed")
        if destination.exists():
            destination.rmdir()  # Only the verified empty destination.
        stage.rename(destination)
