import pytest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient


def _make_app_with_mock_agent():
    """Create app instance with a mocked ParliamentAgent."""
    mock_agent = MagicMock()

    async def mock_chat(*args, **kwargs):
        return {
            "answer": "The committee discussed fiscal sustainability.",
            "sources": ["scottish_parliament_report_07_01_25.txt (Speaker: Stephen Boyle)"],
            "content": ["The Auditor General discussed fiscal sustainability measures..."],
        }

    mock_agent.chat = mock_chat

    with patch("main.ParliamentAgent", return_value=mock_agent):
        import importlib
        import main
        importlib.reload(main)
        client = TestClient(main.app)
        main.agent = mock_agent
        return client, mock_agent


@pytest.fixture
def client_and_agent():
    return _make_app_with_mock_agent()


class TestHealthEndpoint:
    def test_health_check(self, client_and_agent):
        client, _ = client_and_agent
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json() == {"status": "healthy"}


class TestChatEndpoint:
    def test_chat_returns_valid_response(self, client_and_agent):
        client, mock_agent = client_and_agent
        response = client.post(
            "/chat",
            json={"query": "What topics were discussed?"},
        )
        assert response.status_code == 200
        data = response.json()
        assert "response" in data
        assert isinstance(data["response"], str)
        assert len(data["response"]) > 0

    def test_chat_returns_sources(self, client_and_agent):
        client, _ = client_and_agent
        response = client.post(
            "/chat",
            json={"query": "What did Audit Scotland say?"},
        )
        data = response.json()
        assert "sources" in data
        assert data["sources"] is not None
        assert isinstance(data["sources"], list)
        assert len(data["sources"]) > 0

    def test_chat_returns_content(self, client_and_agent):
        client, _ = client_and_agent
        response = client.post(
            "/chat",
            json={"query": "What did Audit Scotland say?"},
        )
        data = response.json()
        assert "content" in data
        assert data["content"] is not None
        assert isinstance(data["content"], list)
        assert len(data["content"]) > 0

    def test_chat_endpoint_returns_422_on_missing_query(self, client_and_agent):
        client, _ = client_and_agent
        response = client.post("/chat", json={})
        assert response.status_code == 422

    def test_chat_endpoint_returns_500_on_agent_error(self, client_and_agent):
        client, mock_agent = client_and_agent

        async def mock_chat_error(*args, **kwargs):
            raise RuntimeError("LLM error")

        mock_agent.chat = mock_chat_error

        response = client.post(
            "/chat",
            json={"query": "test query"},
        )
        assert response.status_code == 500

    def test_chat_endpoint_exists(self, client_and_agent):
        client, _ = client_and_agent
        response = client.post(
            "/chat",
            json={"query": "What topics were discussed?"},
        )
        assert response.status_code != 404


class TestIngestionPipelineConfig:
    """Test that the ingestion pipeline components are importable and configurable."""

    def test_keyword_extractor_importable(self):
        from llama_index.core.extractors import KeywordExtractor
        assert KeywordExtractor is not None

    def test_ingestion_pipeline_importable(self):
        from llama_index.core.ingestion import IngestionPipeline
        assert IngestionPipeline is not None
