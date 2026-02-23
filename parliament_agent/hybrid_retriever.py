import logging
from typing import List

from llama_index.core import QueryBundle
from llama_index.core.schema import NodeWithScore
from llama_index.core.retrievers import BaseRetriever, VectorIndexRetriever
from llama_index.retrievers.bm25 import BM25Retriever

logger = logging.getLogger(__name__)


class HybridRetriever(BaseRetriever):
    """
    Hybrid retriever that performs both semantic search (vector) and keyword search (BM25).

    Interleaves results from both retrievers to get diverse, relevant chunks.
    """

    def __init__(
        self,
        vector_retriever: VectorIndexRetriever,
        bm25_retriever: BM25Retriever,
        mode: str = "interleave",
    ) -> None:
        """
        Initialize hybrid retriever.

        Args:
            vector_retriever: Semantic search retriever
            bm25_retriever: Keyword search retriever
            mode: How to combine results
                - "interleave": Alternate between vector and BM25 results
                - "concat": Vector results first, then BM25 results
        """
        self._vector_retriever = vector_retriever
        self._bm25_retriever = bm25_retriever
        self._mode = mode
        super().__init__()

    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        """
        Retrieve nodes using both vector and BM25 search, then combine.

        Args:
            query_bundle: Query string wrapped in QueryBundle

        Returns:
            Combined list of unique nodes from both retrievers
        """
        logger.info(
            f"Hybrid retrieval for query: {query_bundle.query_str[:100]}...")

        vector_nodes = self._vector_retriever.retrieve(query_bundle)
        bm25_nodes = self._bm25_retriever.retrieve(query_bundle)

        logger.info(f"Vector retriever returned {len(vector_nodes)} nodes")
        logger.info(f"BM25 retriever returned {len(bm25_nodes)} nodes")

        if self._mode == "interleave":
            return self._interleave_results(vector_nodes, bm25_nodes)
        elif self._mode == "concat":
            return self._concat_results(vector_nodes, bm25_nodes)
        else:
            raise ValueError(
                f"Invalid mode: {self._mode}. Use 'interleave' or 'concat'")

    def _interleave_results(
        self,
        vector_nodes: List[NodeWithScore],
        bm25_nodes: List[NodeWithScore]
    ) -> List[NodeWithScore]:
        """
        Interleave results from vector and BM25, removing duplicates.
        Pattern: [V1, B1, V2, B2, V3, B3, ...]
        """
        resulting_nodes = []
        node_ids_added = set()

        max_len = max(len(vector_nodes), len(bm25_nodes))

        for i in range(max_len):
            if i < len(vector_nodes):
                vector_node = vector_nodes[i]
                if vector_node.node.node_id not in node_ids_added:
                    resulting_nodes.append(vector_node)
                    node_ids_added.add(vector_node.node.node_id)

            if i < len(bm25_nodes):
                bm25_node = bm25_nodes[i]
                if bm25_node.node.node_id not in node_ids_added:
                    resulting_nodes.append(bm25_node)
                    node_ids_added.add(bm25_node.node.node_id)

        logger.info(f"Interleaved to {len(resulting_nodes)} unique nodes")
        return resulting_nodes

    def _concat_results(
        self,
        vector_nodes: List[NodeWithScore],
        bm25_nodes: List[NodeWithScore]
    ) -> List[NodeWithScore]:
        """
        Concatenate results: vector first, then BM25 (remove duplicates).
        Pattern: [V1, V2, V3, ..., B1, B2, B3, ...]
        """
        resulting_nodes = []
        node_ids_added = set()

        for node in vector_nodes:
            if node.node.node_id not in node_ids_added:
                resulting_nodes.append(node)
                node_ids_added.add(node.node.node_id)

        for node in bm25_nodes:
            if node.node.node_id not in node_ids_added:
                resulting_nodes.append(node)
                node_ids_added.add(node.node.node_id)

        logger.info(f"Concatenated to {len(resulting_nodes)} unique nodes")
        return resulting_nodes
