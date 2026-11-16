from urllib.parse import urlsplit

import httpx

from banschik.config import Config


class WebSearch:
    def __init__(self, client: httpx.AsyncClient, config: Config):
        self.client, self.config = client, config

    async def search(self, query: str) -> dict:
        if not self.config.search_enabled:
            return {"error": "Web search is disabled"}
        if not isinstance(query, str) or not query.strip() or len(query) > 500:
            return {"error": "query must contain 1–500 characters"}
        try:
            response = await self.client.get(
                f"{self.config.search_url}/search",
                params={"q": query, "format": "json", "language": "auto", "safesearch": 1},
                timeout=self.config.search_timeout,
            )
            response.raise_for_status()
            payload = response.json()
            results = []
            for item in payload.get("results", []):
                url = item.get("url", "")
                if urlsplit(url).scheme not in {"http", "https"}:
                    continue
                results.append(
                    {
                        "title": str(item.get("title", ""))[:200],
                        "url": url[:2000],
                        "snippet": str(item.get("content", ""))[:1200],
                    }
                )
                if len(results) == self.config.search_results:
                    break
            return {"results": results, "note": "Search snippets, not full page contents"}
        except (httpx.HTTPError, ValueError, TypeError):
            return {"error": "Search unavailable. Do not invent search results."}
