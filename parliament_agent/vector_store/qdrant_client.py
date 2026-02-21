"""
Qdrant vector store manager for the Parliamentary Agent.

Handles connection management, collection lifecycle, and provides
both sync and async Qdrant clients for use with FastAPI.
"""
import os
import logging
import time

import yaml
from qdrant_client import QdrantClient, AsyncQdrantClient
from llama_index.vector_stores.qdrant import QdrantVectorStore

logger = logging.getLogger(__name__)

PARAMS_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "params.yaml")


def _load_params() -> dict:
    with open(PARAMS_PATH, "r") as f:
        return yaml.safe_load(f)


class QdrantVectorStoreManager:
    """Manages Qdrant connections and the LlamaIndex QdrantVectorStore."""

    def __init__(
        self,
        host: str | None = None,
        port: int | None = None,
        collection_name: str | None = None,
    ):
        params = _load_params()
        self.host = host or os.getenv("QDRANT_HOST", "localhost")
        self.port = port or int(os.getenv("QDRANT_PORT", "6333"))
        self.collection_name = collection_name or params["COLLECTION_NAME"]

        self._client = self._connect_sync()
        self._aclient = self._connect_async()

        self.vector_store = QdrantVectorStore(
            client=self._client,
            aclient=self._aclient,
            collection_name=self.collection_name,
        )

    def _connect_sync(self) -> QdrantClient:
        max_retries = 5
        for attempt in range(1, max_retries + 1):
            try:
                client = QdrantClient(host=self.host, port=self.port)
                client.get_collections()
                logger.info("Connected to Qdrant at %s:%s", self.host, self.port)
                return client
            except Exception:
                if attempt == max_retries:
                    raise
                wait = 2 ** attempt
                logger.warning(
                    "Qdrant not ready (attempt %d/%d), retrying in %ds...",
                    attempt, max_retries, wait,
                )
                time.sleep(wait)

    def _connect_async(self) -> AsyncQdrantClient:
        return AsyncQdrantClient(host=self.host, port=self.port)

    def collection_exists(self) -> bool:
        """Check whether the collection already has vectors."""
        collections = self._client.get_collections().collections
        for col in collections:
            if col.name == self.collection_name:
                info = self._client.get_collection(self.collection_name)
                return info.points_count > 0
        return False

    @property
    def client(self) -> QdrantClient:
        return self._client

    @property
    def aclient(self) -> AsyncQdrantClient:
        return self._aclient
