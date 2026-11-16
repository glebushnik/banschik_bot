import io
import json
import tarfile
from datetime import datetime, timedelta

import pytest

from banschik.backup import Backups, restore, verify
from banschik.memory import Memory
from banschik.storage import Store
from tests.conftest import fake_embed, save


def test_snapshot_rotation_and_restore(storage, memory, config, tmp_path):
    save(storage)
    memory.index_pending()
    backups = Backups(storage, memory, config)
    (config.backup_dir / "other.txt").write_text("keep")
    now = datetime(2026, 11, 16, 3, tzinfo=config.timezone)
    for offset in range(5):
        archive = backups.create(now + timedelta(days=offset))
    assert len(list(config.backup_dir.glob("banschik-*.tar.gz"))) == 3
    assert (config.backup_dir / "other.txt").exists()
    manifest = verify(archive)
    assert "memory.sqlite3" in manifest["files"]
    destination = tmp_path / "restored"
    restore(archive, destination)
    restored = Store(destination / "memory.sqlite3")
    try:
        assert restored.stats()["messages"] == 1
        from dataclasses import replace

        recalled = Memory(restored, replace(config, data_dir=destination), fake_embed)
        assert recalled.search(-100123, "баня")[0]["text"] == "баня завтра"
    finally:
        restored.close()
    with pytest.raises(ValueError, match="empty"):
        restore(archive, destination)


def test_corrupt_backup_does_not_prune(storage, memory, config, monkeypatch):
    backups = Backups(storage, memory, config)
    old = backups.create(datetime(2026, 11, 1, 3, tzinfo=config.timezone))

    def fail(*args):
        raise ValueError("checksum")

    monkeypatch.setattr("banschik.backup.verify", fail)
    with pytest.raises(ValueError):
        backups.create(datetime(2026, 11, 16, 3, tzinfo=config.timezone))
    assert old.exists()


def test_unsafe_archive_is_rejected(tmp_path):
    archive = tmp_path / "bad.tar.gz"
    manifest = {"format": 1, "files": {"memory.sqlite3": "invalid", "../escape": "invalid"}}
    with tarfile.open(archive, "w:gz") as tar:
        data = json.dumps(manifest).encode()
        info = tarfile.TarInfo("manifest.json")
        info.size = len(data)
        tar.addfile(info, io.BytesIO(data))
        info = tarfile.TarInfo("memory.sqlite3")
        info.size = 0
        tar.addfile(info, io.BytesIO())
    with pytest.raises(ValueError):
        restore(archive, tmp_path / "target")
