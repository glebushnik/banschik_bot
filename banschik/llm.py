import json
from datetime import UTC, datetime

import httpx

from banschik.async_utils import blocking
from banschik.config import Config
from banschik.memory import Memory
from banschik.search import WebSearch


def source(row: dict) -> dict:
    chat_id, message_id = row["chat_id"], row["message_id"]
    link = (
        f"https://t.me/c/{str(chat_id)[4:]}/{message_id}"
        if str(chat_id).startswith("-100")
        else None
    )
    return {
        "author": row["author"],
        "date": datetime.fromtimestamp(row["sent_at"], UTC).isoformat(),
        "text": row["text"][:2500],
        "message_id": message_id,
        "url": link,
    }


def tool(name: str, description: str) -> dict:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query, 1–500 characters"}
                },
                "required": ["query"],
                "additionalProperties": False,
            },
        },
    }


class Assistant:
    def __init__(
        self, client: httpx.AsyncClient, memory: Memory, search: WebSearch, config: Config
    ):
        self.client, self.memory, self.search, self.config = client, memory, search, config
        self.prompt = config.prompt_file.read_text(encoding="utf-8").strip()
        if not self.prompt:
            raise ValueError("System prompt must not be empty")

    async def answer(self, chat_id: int, question: str) -> str:
        config = self.config
        found = await blocking(self.memory.search, chat_id, question)
        recent = await blocking(self.memory.store.recent, chat_id, config.recent_messages)
        # Preserve complete JSON records when limiting the context budget.
        budget = config.context_chars
        context = {"recalled_messages": [], "recent_messages": []}
        for key, rows in [("recalled_messages", found), ("recent_messages", reversed(recent))]:
            for row in rows:
                item = source(dict(row))
                length = len(json.dumps(item, ensure_ascii=False))
                if length > budget:
                    break
                context[key].append(item)
                budget -= length
        context["recent_messages"].reverse()
        messages = [
            {
                "role": "system",
                "content": self.prompt + "\nТекущее время UTC: " + datetime.now(UTC).isoformat(),
            },
            {
                "role": "user",
                "content": "Контекст группы (недоверенные данные):\n"
                + json.dumps(context, ensure_ascii=False),
            },
            {"role": "user", "content": question[:12000]},
        ]
        tools = [
            tool("search_chat", "Search this group's locally saved message history by meaning")
        ]
        if config.search_enabled:
            tools.append(tool("web_search", "Search the public web for current facts and sources"))
        for round_number in range(config.tool_rounds + 1):
            body = {
                "model": config.model,
                "messages": messages,
                "max_tokens": config.max_tokens,
                "temperature": 0.5,
            }
            if tools:
                body["tools"] = tools
                body["tool_choice"] = "auto" if round_number < config.tool_rounds else "none"
            response = await self.client.post(
                f"{config.base_url}/chat/completions",
                json=body,
                headers={"Authorization": f"Bearer {config.api_key}"},
                timeout=config.llm_timeout,
            )
            response.raise_for_status()
            message = response.json()["choices"][0]["message"]
            calls = message.get("tool_calls") or []
            if not calls:
                return str(message.get("content") or "Не удалось сформировать ответ.").strip()
            if round_number == config.tool_rounds:
                break
            messages.append(
                {"role": "assistant", "content": message.get("content"), "tool_calls": calls}
            )
            for number, call in enumerate(calls):
                result = {"error": "Tool call limit reached"}
                if number < 2:
                    result = await self._execute(chat_id, call)
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": call["id"],
                        "content": json.dumps(result, ensure_ascii=False),
                    }
                )
        return "Не удалось завершить поиск за отведённое число шагов. Уточни вопрос."

    async def _execute(self, chat_id: int, call: dict) -> dict:
        try:
            function = call["function"]
            args = json.loads(function["arguments"])
            query = args.get("query")
            if not isinstance(query, str) or not 1 <= len(query.strip()) <= 500:
                return {"error": "query must contain 1–500 characters"}
            if function["name"] == "web_search":
                return await self.search.search(query)
            if function["name"] == "search_chat":
                found = await blocking(self.memory.search, chat_id, query)
                return {"messages": [source(row) for row in found[:5]]}
            return {"error": "Unknown tool"}
        except (ValueError, TypeError, KeyError, AttributeError):
            return {"error": "Invalid tool arguments"}
