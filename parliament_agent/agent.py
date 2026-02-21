"""
Parliamentary Meeting Minutes Conversational Agent.

RAG-based agent using LlamaIndex with Qdrant vector store,
Gemini LLM, and HuggingFace embeddings over Scottish Parliament transcripts.
"""
import os
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional

import yaml
from dotenv import load_dotenv

from llama_index.core import (
    Settings,
    SimpleDirectoryReader,
    VectorStoreIndex,
    StorageContext,
)
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.extractors import KeywordExtractor
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.llms import ChatMessage, MessageRole
from llama_index.llms.google_genai import GoogleGenAI
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

import google.genai.types as types

from parliament_agent.vector_store.qdrant_client import QdrantVectorStoreManager

logger = logging.getLogger(__name__)

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent
PARAMS_PATH = BASE_DIR / "params.yaml"
DATA_DIR = BASE_DIR / "data"


def _load_params() -> dict:
    with open(PARAMS_PATH, "r") as f:
        return yaml.safe_load(f)


class ParliamentAgent:
    """
    Conversational AI agent for querying parliamentary meeting minutes.

    Uses a RAG pipeline: ingest -> chunk -> embed -> store in Qdrant,
    then retrieve + generate with Gemini via condense_plus_context chat mode.
    """

    def __init__(self):
        params = _load_params()

        google_api_key = os.getenv("GOOGLE_API_KEY")
        if not google_api_key:
            raise ValueError("GOOGLE_API_KEY not set in environment")

        config = types.GenerateContentConfig(
            thinking_config=types.ThinkingConfig(thinking_budget=0),
            temperature=0.4
        )

        Settings.llm = GoogleGenAI(
            model=f"models/{params['GENERATION_MODEL']}",
            api_key=google_api_key,
            generation_config=config,
        )

        self.ingest_llm = GoogleGenAI(
            model="models/gemini-2.5-flash-lite",
            api_key=google_api_key,
            generation_config=config,
        )

        Settings.embed_model = HuggingFaceEmbedding(
            model_name=params["EMBEDDING_MODEL"],
        )

        self._qdrant_manager = QdrantVectorStoreManager()

        if self._qdrant_manager.collection_exists():
            logger.info("Collection already populated – loading existing index")
            storage_context = StorageContext.from_defaults(
                vector_store=self._qdrant_manager.vector_store
            )
            self._index = VectorStoreIndex.from_vector_store(
                vector_store=self._qdrant_manager.vector_store,
                storage_context=storage_context,
            )
        else:
            logger.info("Collection empty – running ingestion pipeline")
            self._index = self._ingest()

        self._memory = ChatMemoryBuffer.from_defaults(token_limit=4096)

        self._chat_engine = self._index.as_chat_engine(
            chat_mode="condense_plus_context",
            memory=self._memory,
            llm=Settings.llm,
            verbose=True,
            system_prompt=(
                "You are a helpful assistant that answers questions about "
                "Scottish parliamentary meeting minutes. You MUST answer "
                "based ONLY on the retrieved context. If the information is "
                "not available in the context, say 'I could not find that "
                "information in the parliamentary documents.' "
                "Always be factual and cite which speaker or section the "
                "information comes from when possible."
            ),
        )

        logger.info("ParliamentAgent initialised successfully")

    def _ingest(self) -> VectorStoreIndex:
        """Load documents, chunk with keyword extraction, and index into Qdrant."""
        documents = SimpleDirectoryReader(
            input_dir=str(DATA_DIR),
            filename_as_id=True,
        ).load_data()
        logger.info("Loaded %d documents from %s", len(documents), DATA_DIR)

        Settings.text_splitter = SentenceSplitter(
            chunk_size=400, chunk_overlap=100
        )

        pipeline = IngestionPipeline(
            transformations=[
                Settings.text_splitter,
                KeywordExtractor(keywords=10, llm=self.ingest_llm, num_workers=2),
                Settings.embed_model,
            ],
            vector_store=self._qdrant_manager.vector_store,
        )

        nodes = pipeline.run(documents=documents, show_progress=True, num_workers=2)
        logger.info("Ingestion complete – %d nodes indexed", len(nodes))

        index = VectorStoreIndex.from_vector_store(
            vector_store=self._qdrant_manager.vector_store,
        )
        return index

    async def chat(
        self, query: str, history: Optional[List[Dict[str, str]]] = None
    ) -> Dict[str, Any]:
        """
        Process a user query and return a grounded response.

        Uses retrieval + LLM to answer questions about parliamentary minutes.
        """
        # Build context from history
        context_messages = []
        if history:
            for msg in history:
                role = (
                    MessageRole.USER
                    if msg.get("role") == "user"
                    else MessageRole.ASSISTANT
                )
                context_messages.append(ChatMessage(role=role, content=msg["content"]))
        
        # Retrieve relevant documents
        retriever = self._index.as_retriever(similarity_top_k=5)
        nodes = await retriever.aretrieve(query)
        
        # Build context from retrieved nodes
        context_str = "\n\n".join([
            f"Document {i+1}:\n{node.get_content()}"
            for i, node in enumerate(nodes)
        ])
        
        # Build prompt with history and context
        system_prompt = (
            "You are a helpful assistant that answers questions about "
            "Scottish parliamentary meeting minutes. You MUST answer "
            "based ONLY on the retrieved context. If the information is "
            "not available in the context, say 'I could not find that "
            "information in the parliamentary documents.' "
            "Always be factual and cite which speaker or section the "
            "information comes from when possible."
        )
        
        # Add history and current query
        messages = [ChatMessage(role=MessageRole.SYSTEM, content=system_prompt)]
        messages.extend(context_messages)
        messages.append(ChatMessage(
            role=MessageRole.USER,
            content=f"Context:\n{context_str}\n\nQuestion: {query}"
        ))
        
        # Get response from LLM
        response = await Settings.llm.achat(messages)
        
        # Extract sources
        sources = []
        for node in nodes:
            source_info = node.metadata.get("file_name", "unknown")
            keywords = node.metadata.get("excerpt_keywords", "")
            if keywords:
                source_info = f"{source_info} (Keywords: {keywords})"
            sources.append(source_info)

        return {"answer": str(response.message.content), "sources": sources}

    def _retrieve(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Retrieve relevant document chunks for a query."""
        retriever = self._index.as_retriever(similarity_top_k=top_k)
        nodes = retriever.retrieve(query)
        return [
            {
                "text": node.get_content(),
                "score": node.get_score(),
                "metadata": node.metadata,
            }
            for node in nodes
        ]
