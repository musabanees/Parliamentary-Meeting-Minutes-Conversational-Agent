"""
Parliamentary Meeting Minutes Conversational Agent.

RAG-based agent using LlamaIndex with Qdrant vector store,
Gemini LLM, and HuggingFace embeddings over Scottish Parliament transcripts.
"""
import os
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime

import yaml
from dotenv import load_dotenv

from llama_index.core import (
    Settings,
    SimpleDirectoryReader,
    VectorStoreIndex,
    StorageContext,
    get_response_synthesizer,
)
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.extractors import KeywordExtractor
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.callbacks import CallbackManager, LlamaDebugHandler
# from llama_index.postprocessor.flag_embedding_reranker import FlagEmbeddingReranker  # BGE reranker
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.chat_engine import CondensePlusContextChatEngine
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

import google.genai.types as types

from parliament_agent.vector_store.qdrant_client import QdrantVectorStoreManager
from parliament_agent.retrievers import HybridRetriever

logger = logging.getLogger(__name__)

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

BASE_DIR = Path(__file__).resolve().parent.parent
PARAMS_PATH = BASE_DIR / "params.yaml"
DATA_DIR = BASE_DIR / "data"
LOGS_DIR = BASE_DIR / "logs"

# Configure logging to save to logs folder
LOGS_DIR.mkdir(exist_ok=True)
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
log_file = LOGS_DIR / f"agent_{timestamp}.log"

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ],
    force=True
)
logger.info(f"Agent logging to {log_file}")


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
        generation_model = params['GENERATION_MODEL']
        # judge_model = params['JUDGE_MODEL']
        # llm_judge = OpenAI(
        #     model=judge_model,
        #     api_key=OPENAI_API_KEY,
        #     temperature=0.4,
        # )
        Settings.llm = OpenAI(
            model=generation_model,
            api_key=OPENAI_API_KEY,
            temperature=0.3,
        )

        # Initialize LlamaDebugHandler for tracking timing
        llama_debug = LlamaDebugHandler(print_trace_on_end=True)
        Settings.callback_manager = CallbackManager([llama_debug])
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

        # Set up Hybrid Retriever (BM25 + Vector Search)
        logger.info("Setting up hybrid retriever (BM25 + Vector)...")

        # Get all nodes from the index docstore for BM25
        # The docstore contains RefDocNode objects, we need to get the actual nodes
        all_nodes = []
        for node_id, node in self._index.docstore.docs.items():
            if hasattr(node, 'node'):
                all_nodes.append(node.node)
            else:
                all_nodes.append(node)

        logger.info(f"Retrieved {len(all_nodes)} nodes for BM25 indexing")

        if len(all_nodes) == 0:
            # Fallback: retrieve nodes via the retriever
            temp_retriever = self._index.as_retriever(similarity_top_k=1000)
            all_nodes = temp_retriever.retrieve("parliament")
            all_nodes = [n.node for n in all_nodes]
            logger.info(f"Fallback: Retrieved {len(all_nodes)} nodes via retriever")

        # Create BM25 retriever
        bm25_retriever = BM25Retriever.from_defaults(
            nodes=all_nodes,
            similarity_top_k=5,  # Reduced from 13 since no reranking
        )

        # Create vector retriever
        vector_retriever = VectorIndexRetriever(
            index=self._index,
            similarity_top_k=5,  # Reduced from 13 since no reranking
        )

        # Create hybrid retriever
        hybrid_retriever = HybridRetriever(
            vector_retriever=vector_retriever,
            bm25_retriever=bm25_retriever,
            mode="interleave",  # Alternate between vector and BM25 results
        )

        # # Initialize BGE reranker to get top 5 chunks after hybrid retrieval
        # logger.info("Initializing BGE reranker (BAAI/bge-reranker-v2-m3)...")
        # bge_reranker = FlagEmbeddingReranker(
        #     model="BAAI/bge-reranker-v2-m3",
        #     top_n=5,
        #     use_fp16=True,
        # )

        # Wrap in CondensePlusContextChatEngine for conversational context
        self._chat_engine = CondensePlusContextChatEngine.from_defaults(
            retriever=hybrid_retriever,
            memory=self._memory,
            llm=Settings.llm,
            verbose=True,
            # node_postprocessors=[bge_reranker],  # BGE reranker commented out
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
                KeywordExtractor(keywords=10, llm=Settings.llm, num_workers=2),
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

    async def chat(self, query: str) -> Dict[str, Any]:

        # 1. Display/Log current Memory state before the query
        current_memory = self._memory.get_all()
        if current_memory:
            logger.info(f"Total messages in memory: {len(current_memory)}")
            for idx, msg in enumerate(current_memory, 1):
                content = str(msg.content) if msg.content else ""
                logger.info(f"  [{idx}] {msg.role.value.upper()}: {content[:150]}...")
        else:
            logger.info("Memory is empty (first turn in conversation)")

        logger.info(f"--- [NEW USER QUERY] ---")
        logger.info(f"Original Query: {query}")

        response = await self._chat_engine.achat(query)

        logger.info("--- [QUERY REWRITING COMPLETE] ---")
        if not current_memory:
            logger.info("Note: First query in conversation - likely used as-is")
        else:
            logger.info("Note: Query was contextualized based on conversation history")

        # LlamaIndex can expose sources via response.source_nodes
        logger.info("--- [RETRIEVAL SCORES] ---")
        sources = []
        content_list = []
        score_list = []
        if hasattr(response, "source_nodes") and response.source_nodes:
            logger.info(f"Retrieved {len(response.source_nodes)} relevant chunks:")
            for idx, n in enumerate(response.source_nodes, 1):
                score = n.get_score() if hasattr(n, 'get_score') else 0.0
                content = n.get_content() if hasattr(n, 'get_content') else str(n.node.get_content())
                fn = n.metadata.get("file_name", "unknown") if hasattr(n, 'metadata') else n.node.metadata.get("file_name", "unknown")
                kw = n.metadata.get("excerpt_keywords", "") if hasattr(n, 'metadata') else n.node.metadata.get("excerpt_keywords", "")

                sources.append(f"{fn} (Keywords: {kw})" if kw else fn)
                content_list.append(content)
                score_list.append(score)
                # logger.info(f"  Score: {score:.4f} | Content: {content[:150]}...")
        else:
            logger.info("No source nodes retrieved")

        updated_memory = self._memory.get_all()
        logger.info(f"Memory now contains {len(updated_memory)} messages")

        # return {"answer": str(response), "sources": list(set(sources)), "content": content}
        return {"answer": str(response),
                "content": content_list,
                "sources": sources}


    def _retrieve(self, query: str, top_k: int = 8) -> List[Dict[str, Any]]:
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
