import os
import logging
from pathlib import Path
from typing import List, Dict, Any
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
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.callbacks import CallbackManager, LlamaDebugHandler
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.postprocessor.rankGPT_rerank import RankGPTRerank
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.chat_engine import CondensePlusContextChatEngine
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding

from parliament_agent.qdrant_client import QdrantVectorStoreManager
from parliament_agent.hybrid_retriever import HybridRetriever

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
        self.params = _load_params()
        self.generation_model = self.params['GENERATION_MODEL']
        self.judge_model = self.params['JUDGE_MODEL']
        self.llm_judge = OpenAI(
            model=self.judge_model,
            api_key=OPENAI_API_KEY,
            temperature=0.4,
        )

        Settings.llm = OpenAI(
            model=self.generation_model,
            api_key=OPENAI_API_KEY,
            temperature=0.3,
        )
        # Initialize LlamaDebugHandler for tracking timing
        llama_debug = LlamaDebugHandler(print_trace_on_end=True)
        Settings.callback_manager = CallbackManager([llama_debug])
        Settings.embed_model = OpenAIEmbedding(
            model=self.params["EMBEDDING_MODEL"],
            api_key=OPENAI_API_KEY,
        )

        self._qdrant_manager = QdrantVectorStoreManager()
        if self._qdrant_manager.collection_exists():
            logger.info(
                "Collection already populated – loading existing index")
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

        all_nodes = []
        for node_id, node in self._index.docstore.docs.items():
            if hasattr(node, 'node'):
                all_nodes.append(node.node)
            else:
                all_nodes.append(node)

        logger.info(f"Retrieved {len(all_nodes)} nodes for BM25 indexing")

        if len(all_nodes) == 0:
            temp_retriever = self._index.as_retriever(similarity_top_k=1000)
            all_nodes = temp_retriever.retrieve("parliament")
            all_nodes = [n.node for n in all_nodes]
            logger.info(
                f"Fallback: Retrieved {len(all_nodes)} nodes via retriever")

        bm25_retriever = BM25Retriever.from_defaults(
            nodes=all_nodes,
            similarity_top_k=6,
        )

        vector_retriever = VectorIndexRetriever(
            index=self._index,
            similarity_top_k=10,
        )

        self._hybrid_retriever = HybridRetriever(
            vector_retriever=vector_retriever,
            bm25_retriever=bm25_retriever,
            mode="interleave",
        )

        self._rankgpt = RankGPTRerank(
            top_n=5,
            llm=self.llm_judge,
            verbose=True,
        )

        # Wrap in CondensePlusContextChatEngine for conversational context
        self._chat_engine = CondensePlusContextChatEngine.from_defaults(
            retriever=self._hybrid_retriever,
            memory=self._memory,
            llm=Settings.llm,
            verbose=True,
            node_postprocessors=[self._rankgpt],
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

    def get_query_engine(self, response_mode: str = "compact") -> RetrieverQueryEngine:
        """Return a query engine using the same hybrid retriever + RankGPT reranker."""
        return RetrieverQueryEngine.from_args(
            retriever=self._hybrid_retriever,
            node_postprocessors=[self._rankgpt],
            response_mode=response_mode,
        )

    def _ingest(self) -> VectorStoreIndex:
        """Load documents, apply recursive chunking, and index into Qdrant."""
        documents = SimpleDirectoryReader(
            input_dir=str(DATA_DIR),
            filename_as_id=True,
        ).load_data()
        logger.info("Loaded %d documents from %s", len(documents), DATA_DIR)

        # Recursive chunking: tries separators in order [\n\n, \n, " ", ""]
        node_parser = SentenceSplitter(
            chunk_size=500,
            chunk_overlap=120,
            separator=" ",
            paragraph_separator="\n\n",
            secondary_chunking_regex="[^,.;。？！]+[,.;。？！]?",
        )

        nodes = node_parser.get_nodes_from_documents(
            documents, show_progress=True)
        logger.info(
            "Created %d nodes with recursive chunking (500/120)", len(nodes))

        storage_context = StorageContext.from_defaults(
            vector_store=self._qdrant_manager.vector_store
        )

        index = VectorStoreIndex(
            nodes,
            storage_context=storage_context,
            show_progress=True,
        )
        logger.info("Ingestion complete – %d nodes indexed", len(nodes))
        return index

    async def chat(self, query: str) -> Dict[str, Any]:

        # 1. Display/Log current Memory state before the query
        current_memory = self._memory.get_all()
        if current_memory:
            logger.info(f"Total messages in memory: {len(current_memory)}")
            for idx, msg in enumerate(current_memory, 1):
                content = str(msg.content) if msg.content else ""
                logger.info(
                    f"  [{idx}] {msg.role.value.upper()}: {content[:150]}...")
        else:
            logger.info("Memory is empty (first turn in conversation)")

        logger.info(f"--- [NEW USER QUERY] ---")
        logger.info(f"Original Query: {query}")

        response = await self._chat_engine.achat(query)

        logger.info("--- [QUERY REWRITING COMPLETE] ---")
        if not current_memory:
            logger.info(
                "Note: First query in conversation - likely used as-is")
        else:
            logger.info(
                "Note: Query was contextualized based on conversation history")

        # LlamaIndex can expose sources via response.source_nodes
        logger.info("--- [RETRIEVAL SCORES] ---")
        sources = []
        content_list = []
        score_list = []
        if hasattr(response, "source_nodes") and response.source_nodes:
            logger.info(
                f"Retrieved {len(response.source_nodes)} relevant chunks:")
            for idx, n in enumerate(response.source_nodes, 1):
                score = n.get_score() if hasattr(n, 'get_score') else 0.0
                content = n.get_content() if hasattr(
                    n, 'get_content') else str(n.node.get_content())
                fn = n.metadata.get("file_name", "unknown") if hasattr(
                    n, 'metadata') else n.node.metadata.get("file_name", "unknown")
                kw = n.metadata.get("excerpt_keywords", "") if hasattr(
                    n, 'metadata') else n.node.metadata.get("excerpt_keywords", "")

                sources.append(f"{fn} (Keywords: {kw})" if kw else fn)
                content_list.append(content)
                score_list.append(score)
        else:
            logger.info("No source nodes retrieved")

        updated_memory = self._memory.get_all()
        logger.info(f"Memory now contains {len(updated_memory)} messages")

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
