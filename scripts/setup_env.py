"""Create a local configuration without overwriting existing credentials."""

import os
import secrets
from pathlib import Path

root = Path(__file__).resolve().parent.parent
destination = root / ".env"
if destination.exists():
    raise SystemExit(".env already exists; leaving it unchanged")
content = (root / ".env.example").read_text()
content += f"\nSEARXNG_SECRET={secrets.token_hex(32)}\n"
descriptor = os.open(destination, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
with os.fdopen(descriptor, "w") as stream:
    stream.write(content)
print("Created .env. Fill TELEGRAM_BOT_TOKEN, LLM_API_KEY and ALLOWED_CHAT_IDS.")
