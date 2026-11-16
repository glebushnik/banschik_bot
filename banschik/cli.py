import argparse
import asyncio
import logging
import signal
from pathlib import Path

from filelock import FileLock

from banschik.backup import Backups, restore, verify
from banschik.config import Config
from banschik.memory import LocalEmbedder, Memory
from banschik.storage import Store


def main():
    parser = argparse.ArgumentParser(description="Telegram group assistant")
    sub = parser.add_subparsers(dest="command", required=True)
    for command in ("run", "backup", "reindex", "download-model", "stats"):
        sub.add_parser(command)
    checker = sub.add_parser("verify-backup")
    checker.add_argument("archive", type=Path)
    restorer = sub.add_parser("restore")
    restorer.add_argument("archive", type=Path)
    restorer.add_argument("--to", type=Path, required=True)
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    # Request logs would contain Telegram's token in the URL.
    logging.getLogger("httpx").setLevel(logging.CRITICAL)
    logging.getLogger("httpcore").setLevel(logging.CRITICAL)
    try:
        if args.command == "verify-backup":
            manifest = verify(args.archive)
            print(f"Verified backup: {manifest['created_at']}")
            return
        if args.command == "restore":
            restore(args.archive, args.to)
            print(f"Restored to {args.to}")
            return
        config = Config.load(require_credentials=args.command == "run")
        if args.command == "run":
            from banschik.app import run

            async def serve():
                task = asyncio.current_task()
                loop = asyncio.get_running_loop()
                try:
                    loop.add_signal_handler(signal.SIGTERM, task.cancel)
                except NotImplementedError:
                    pass
                await run(config)

            asyncio.run(serve())
        elif args.command == "download-model":
            embedder = LocalEmbedder(config)
            dimension = embedder(["проверка"]).shape[1]
            print(f"Model ready: {config.embedding_model}; dimension={dimension}")
        else:
            config.data_dir.mkdir(parents=True, exist_ok=True)
            with FileLock(str(config.data_dir / "runtime.lock"), timeout=0):
                store = Store(config.data_dir / "memory.sqlite3")
                try:
                    if args.command == "reindex":
                        with store.lock, store.db:
                            store.db.execute("DELETE FROM chunks")
                            store.db.execute("UPDATE messages SET indexed_hash=NULL")
                            store.db.execute("DELETE FROM state WHERE key='embedding_signature'")
                        memory = Memory(store, config)
                        while memory.index_pending():
                            pass
                        memory.flush()
                        print("Reindex complete")
                    elif args.command == "backup":
                        memory = Memory(store, config)
                        print(Backups(store, memory, config).create())
                    else:
                        print(store.stats())
                finally:
                    store.close()
    except (KeyboardInterrupt, asyncio.CancelledError):
        pass
    except Exception as exc:
        # Avoid dumping HTTP request URLs or credentials in tracebacks.
        logging.error(
            "Command failed (%s). Check configuration, network and service logs.",
            type(exc).__name__,
        )
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()
