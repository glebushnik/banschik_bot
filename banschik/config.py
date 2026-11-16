import os
from dataclasses import dataclass, field
from pathlib import Path
from zoneinfo import ZoneInfo

from dotenv import load_dotenv


def integer(name: str, default: int, minimum: int = 1, maximum: int = 1_000_000) -> int:
    value = int(os.getenv(name, str(default)))
    if not minimum <= value <= maximum:
        raise ValueError(f"{name} must be between {minimum} and {maximum}")
    return value


@dataclass(frozen=True)
class Config:
    token: str = field(repr=False)
    api_key: str = field(repr=False)
    allowed_chats: frozenset[int]
    model: str
    base_url: str
    data_dir: Path
    cache_dir: Path
    backup_dir: Path
    embedding_model: str
    embedding_threads: int
    batch_size: int
    index_interval: int
    chunk_size: int
    overlap: int
    top_k: int
    recent_messages: int
    context_chars: int
    llm_timeout: int
    max_tokens: int
    tool_rounds: int
    concurrency: int
    cooldown: int
    search_enabled: bool
    search_url: str
    search_timeout: int
    search_results: int
    timezone: ZoneInfo
    backup_hour: int
    backup_minute: int
    retention_days: int
    prompt_file: Path

    @classmethod
    def load(cls, require_credentials: bool = True) -> "Config":
        load_dotenv()
        token = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
        api_key = os.getenv("LLM_API_KEY", "").strip()
        chats = frozenset(
            int(x.strip()) for x in os.getenv("ALLOWED_CHAT_IDS", "").split(",") if x.strip()
        )
        if require_credentials and (not token or not api_key or not chats):
            raise ValueError("Set TELEGRAM_BOT_TOKEN, LLM_API_KEY and ALLOWED_CHAT_IDS in .env")
        size = integer("CHUNK_SIZE", 700, 100, 1200)
        overlap = integer("CHUNK_OVERLAP", 100, 0, size - 1)
        return cls(
            token=token,
            api_key=api_key,
            allowed_chats=chats,
            model=os.getenv("LLM_MODEL", "google/gemini-2.5-flash"),
            base_url=os.getenv("LLM_BASE_URL", "https://openrouter.ai/api/v1").rstrip("/"),
            data_dir=Path(os.getenv("DATA_DIR", "./data")),
            cache_dir=Path(os.getenv("MODEL_CACHE_DIR", "./data/models")),
            backup_dir=Path(os.getenv("BACKUP_DIR", "./backups")),
            embedding_model=os.getenv("EMBEDDING_MODEL", "intfloat/multilingual-e5-small"),
            embedding_threads=integer("EMBEDDING_THREADS", 2, 1, 64),
            batch_size=integer("EMBEDDING_BATCH_SIZE", 16, 1, 128),
            index_interval=integer("INDEX_INTERVAL_SECONDS", 2, 1, 60),
            chunk_size=size,
            overlap=overlap,
            top_k=integer("RETRIEVAL_TOP_K", 8, 1, 30),
            recent_messages=integer("RECENT_MESSAGES", 20, 0, 100),
            context_chars=integer("CONTEXT_MAX_CHARS", 24000, 1000, 100000),
            llm_timeout=integer("LLM_TIMEOUT_SECONDS", 90, 5, 300),
            max_tokens=integer("LLM_MAX_TOKENS", 1800, 100, 16000),
            tool_rounds=integer("LLM_MAX_TOOL_ROUNDS", 3, 0, 5),
            concurrency=integer("LLM_CONCURRENCY", 2, 1, 8),
            cooldown=integer("REPLY_COOLDOWN_SECONDS", 5, 0, 3600),
            search_enabled=os.getenv("WEB_SEARCH_ENABLED", "true").lower() in {"true", "1"},
            search_url=os.getenv("SEARXNG_URL", "http://127.0.0.1:8080").rstrip("/"),
            search_timeout=integer("WEB_SEARCH_TIMEOUT_SECONDS", 15, 1, 60),
            search_results=integer("WEB_SEARCH_MAX_RESULTS", 5, 1, 10),
            timezone=ZoneInfo(os.getenv("BACKUP_TIMEZONE", "Europe/Moscow")),
            backup_hour=integer("BACKUP_HOUR", 3, 0, 23),
            backup_minute=integer("BACKUP_MINUTE", 0, 0, 59),
            retention_days=integer("BACKUP_RETENTION_DAYS", 3, 1, 365),
            prompt_file=Path(os.getenv("SYSTEM_PROMPT_FILE", "./prompts/system.txt")),
        )
