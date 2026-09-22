import json
import httpx

from adaptive_rag.generator import OllamaGenerator
from adaptive_rag.models import DocumentChunk, SearchHit


def test_ollama_contract_with_mock_transport():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/chat"
        payload = json.loads(request.content)
        assert payload["stream"] is False
        assert "Answer only from the supplied sources" in payload["messages"][0]["content"]
        return httpx.Response(200, json={"message": {"content": "Grounded answer [1]"}})

    transport = httpx.MockTransport(handler)
    gen = OllamaGenerator(base_url="http://ollama.test", model="demo", transport=transport)
    hit = SearchHit(DocumentChunk("1", "Evidence text", "evidence.txt", 0), 1.0, "lexical")
    assert gen.generate("Question?", [hit]) == "Grounded answer [1]"
