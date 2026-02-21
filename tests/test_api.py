"""
API and agent tests for the Parliamentary Minutes Chat API.
"""
import pytest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient


def _make_app_with_mock_agent():
    """Create app instance with a mocked ParliamentAgent."""
    mock_agent = MagicMock()
    
    # Make chat return an awaitable
    async def mock_chat(*args, **kwargs):
        return {
            "answer": "The committee discussed fiscal sustainability.",
            "sources": ["scottish_parliament_report_07_01_25.txt (Speaker: Stephen Boyle)"],
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
            json={"query": "What topics were discussed?", "history": []},
        )
        assert response.status_code == 200
        data = response.json()
        assert "response" in data
        assert "sources" in data
        assert isinstance(data["response"], str)
        assert len(data["response"]) > 0

    def test_chat_with_history(self, client_and_agent):
        client, mock_agent = client_and_agent
        response = client.post(
            "/chat",
            json={
                "query": "What did he say next?",
                "history": [
                    {"role": "user", "content": "What did Stephen Boyle discuss?"},
                    {"role": "assistant", "content": "He discussed fiscal sustainability."},
                ],
            },
        )
        assert response.status_code == 200
        # Verify response includes expected keys
        data = response.json()
        assert "response" in data
        assert "sources" in data

    def test_chat_without_history(self, client_and_agent):
        client, mock_agent = client_and_agent
        response = client.post(
            "/chat",
            json={"query": "Who is the Convener?"},
        )
        assert response.status_code == 200
        data = response.json()
        assert "response" in data

    def test_chat_sources_in_response(self, client_and_agent):
        client, _ = client_and_agent
        response = client.post(
            "/chat",
            json={"query": "What did Audit Scotland say?", "history": []},
        )
        data = response.json()
        assert data["sources"] is not None
        assert isinstance(data["sources"], list)

    def test_chat_endpoint_returns_422_on_missing_query(self, client_and_agent):
        client, _ = client_and_agent
        response = client.post("/chat", json={"history": []})
        assert response.status_code == 422

    def test_chat_endpoint_returns_500_on_agent_error(self, client_and_agent):
        client, mock_agent = client_and_agent
        
        # Make chat raise an error
        async def mock_chat_error(*args, **kwargs):
            raise RuntimeError("LLM error")
        
        mock_agent.chat = mock_chat_error
        
        response = client.post(
            "/chat",
            json={"query": "test query", "history": []},
        )
        assert response.status_code == 500

    def test_chat_endpoint_exists(self, client_and_agent):
        client, _ = client_and_agent
        response = client.post(
            "/chat",
            json={"query": "What topics were discussed?", "history": []},
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
