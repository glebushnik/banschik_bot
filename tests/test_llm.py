import json

import httpx

from banschik.llm import Assistant
from banschik.search import WebSearch
from tests.conftest import save


async def test_tool_loop_uses_local_search_and_sends_system_prompt(memory, config):
    bodies = []

    def respond(request):
        if request.url.path.endswith("/search"):
            return httpx.Response(
                200,
                json={
                    "results": [
                        {
                            "title": "Бани",
                            "url": "https://example.com/banya",
                            "content": "Расписание",
                        }
                    ]
                },
            )
        body = json.loads(request.content)
        bodies.append(body)
        if len(bodies) == 1:
            msg = {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call-1",
                        "type": "function",
                        "function": {
                            "name": "web_search",
                            "arguments": '{"query":"расписание бань"}',
                        },
                    }
                ],
            }
        else:
            msg = {"role": "assistant", "content": "Расписание: https://example.com/banya"}
        return httpx.Response(200, json={"choices": [{"message": msg}]})

    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
        assistant = Assistant(client, memory, WebSearch(client, config), config)
        answer = await assistant.answer(-100123, "Найди расписание")
    assert "https://example.com/banya" in answer
    assert "Не используй смайлики" in bodies[0]["messages"][0]["content"]
    tool_message = bodies[1]["messages"][-1]
    assert tool_message["tool_call_id"] == "call-1"
    assert "Расписание" in tool_message["content"]


async def test_history_tool_cannot_choose_another_chat(storage, memory, config):
    save(storage, "секрет", chat_id=-999)
    save(storage, "баня", update_id=2)
    memory.index_pending()
    async with httpx.AsyncClient() as client:
        assistant = Assistant(client, memory, WebSearch(client, config), config)
        result = await assistant._execute(
            -100123,
            {
                "function": {
                    "name": "search_chat",
                    "arguments": '{"query":"секрет", "chat_id":-999}',
                }
            },
        )
    assert len(result["messages"]) == 1
    assert result["messages"][0]["text"] == "баня"


async def test_search_failure_is_a_tool_result(memory, config):
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(lambda request: httpx.Response(503))
    ) as client:
        result = await WebSearch(client, config).search("баня")
    assert "error" in result


async def test_tool_budget_forces_final_response(memory, config):
    from dataclasses import replace

    config = replace(config, tool_rounds=1)
    calls = []

    def respond(request):
        body = json.loads(request.content)
        calls.append(body)
        if len(calls) == 1:
            message = {
                "content": None,
                "tool_calls": [
                    {
                        "id": "a",
                        "type": "function",
                        "function": {"name": "search_chat", "arguments": '{"query":"баня"}'},
                    }
                ],
            }
        else:
            assert body["tool_choice"] == "none"
            message = {"content": "Сведений пока нет."}
        return httpx.Response(200, json={"choices": [{"message": message}]})

    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
        answer = await Assistant(client, memory, WebSearch(client, config), config).answer(
            -100123, "когда баня?"
        )
    assert answer == "Сведений пока нет."
    assert len(calls) == 2
