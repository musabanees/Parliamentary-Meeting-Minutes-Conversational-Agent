"""
Basic API tests for the Parliamentary Minutes Chat API.

These tests verify that the API endpoints are working correctly.
"""
import pytest
from fastapi.testclient import TestClient
from main import app

client = TestClient(app)


def test_health_check():
    """Test that the health check endpoint returns successfully."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}


def test_chat_endpoint_exists():
    """Test that the chat endpoint exists and accepts POST requests."""
    response = client.post(
        "/chat",
        json={"query": "What topics were discussed?", "history": []},
    )
    # The endpoint should exist (not 404)
    assert response.status_code != 404


# TODO: Add more comprehensive tests for your agent implementation
# Examples:
# - Test that the chat endpoint returns expected response format
# - Test conversation history handling
# - Test error handling for invalid inputs
# - Test that responses are grounded in the documents
