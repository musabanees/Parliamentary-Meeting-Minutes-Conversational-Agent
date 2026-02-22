"""
Evaluation script for the Parliamentary RAG system.

Generates question-answer pairs from indexed nodes and evaluates
retriever performance with different top_k values.
"""
import asyncio
import yaml
import re
import uuid
import warnings
import time
import logging
from typing import List, Dict, Any
from pathlib import Path
from datetime import datetime

import pandas as pd
from tqdm import tqdm

from llama_index.core import Settings
from llama_index.core.schema import MetadataMode, TextNode
from llama_index.core.evaluation import EmbeddingQAFinetuneDataset, RetrieverEvaluator

from parliament_agent.agent import ParliamentAgent


class ParliamentaryEvaluator:
    """Handles evaluation of the Parliamentary RAG system."""
    
    DEFAULT_QA_GENERATE_PROMPT_TMPL = """\
Context information is below.

---------------------
{context_str}
---------------------

Given the context information and not prior knowledge.
generate only questions based on the below query.

You are a Teacher/ Professor. Your task is to setup \
{num_questions_per_chunk} questions for an upcoming \
quiz/examination. The questions should be diverse in nature \
across the document. Restrict the questions to the \
context information provided."
"""
    
    def __init__(self, params_path: str = "params.yaml"):
        """Initialize evaluator with configuration from params.yaml."""
        with open(params_path, "r") as f:
            self.params = yaml.safe_load(f)
        
        self.run_name = self.params["EVALUATION_RUN_NAME"]
        self.run_description = self.params.get("EVALUATION_RUN_DESCRIPTION", "")
        self.dataset_path = Path(self.params["EVALUATION_DATASET_PATH"])
        self.results_base_path = Path(self.params["EVALUATION_RESULTS_PATH"])
        self.top_k_values = self.params["TOP_K_VALUES"]
        self.num_questions_per_chunk = self.params["NUM_QUESTIONS_PER_CHUNK"]
        self.request_delay = self.params["REQUEST_DELAY"]
        self.max_nodes = self.params.get("MAX_NODES_FOR_EVAL")
        
        # Setup logging
        self._setup_logging()
        
        # Results path (no separate folder)
        self.results_base_path.mkdir(parents=True, exist_ok=True)
        
        logging.info(f"Initialized evaluator for run: {self.run_name}")
    
    def _setup_logging(self):
        """Setup logging to both console and file."""
        logs_dir = Path("logs")
        logs_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        log_file = logs_dir / f"evaluation_{timestamp}.log"
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        logging.info(f"Logging to {log_file}")
    
    def generate_questions(
        self,
        nodes: List[TextNode],
        llm,
        qa_generate_prompt_tmpl: str = None
    ) -> EmbeddingQAFinetuneDataset:
        """Generate question-answer pairs from nodes."""
        if qa_generate_prompt_tmpl is None:
            qa_generate_prompt_tmpl = self.DEFAULT_QA_GENERATE_PROMPT_TMPL
        
        node_dict = {
            node.node_id: node.get_content(metadata_mode=MetadataMode.NONE)
            for node in nodes
        }

        queries = {}
        relevant_docs = {}

        logging.info(f"Generating {self.num_questions_per_chunk} questions per node...")
        
        for node_id, text in tqdm(node_dict.items(), desc="Generating questions"):
            query = qa_generate_prompt_tmpl.format(
                context_str=text, 
                num_questions_per_chunk=self.num_questions_per_chunk
            )
            
            try:
                response = llm.complete(query)
                result = str(response).strip().split("\n")
                questions = [
                    re.sub(r"^\d+[\).\s]", "", question).strip() 
                    for question in result
                ]
                questions = [q for q in questions if len(q) > 0][
                    :self.num_questions_per_chunk
                ]

                num_questions_generated = len(questions)
                if num_questions_generated < self.num_questions_per_chunk:
                    logging.warning(
                        f"Node {node_id}: Generated {num_questions_generated} "
                        f"questions, expected {self.num_questions_per_chunk}"
                    )

                for question in questions:
                    question_id = str(uuid.uuid4())
                    queries[question_id] = question
                    relevant_docs[question_id] = [node_id]

                time.sleep(self.request_delay)
                
            except Exception as e:
                logging.error(f"Error generating questions for node {node_id}: {e}")
                continue

        logging.info(f"Generated {len(queries)} total questions from {len(nodes)} nodes")
        
        return EmbeddingQAFinetuneDataset(
            queries=queries, corpus=node_dict, relevant_docs=relevant_docs
        )
    
    def evaluate_retriever(
        self, 
        agent: ParliamentAgent, 
        eval_dataset: EmbeddingQAFinetuneDataset
    ) -> pd.DataFrame:
        """Evaluate retriever with different TOP_K values."""
        all_results = []
        
        logging.info(f"Evaluating with TOP_K values: {self.top_k_values}")
        
        for k in self.top_k_values:
            logging.info(f"Evaluating with top_k={k}...")
            retriever = agent._index.as_retriever(similarity_top_k=k)
            
            retriever_evaluator = RetrieverEvaluator.from_metric_names(
                ["mrr", "hit_rate"], retriever=retriever
            )
            
            import nest_asyncio
            nest_asyncio.apply()
            
            eval_results = asyncio.run(retriever_evaluator.aevaluate_dataset(
                eval_dataset, workers=2
            ))
            
            result_df = self._format_results(f"top_{k}", eval_results)
            all_results.append(result_df)
            logging.info(f"top_{k}: Hit Rate={result_df['Hit Rate'].values[0]:.2f}, MRR={result_df['MRR'].values[0]:.3f}")
        
        summary_df = pd.concat(all_results, ignore_index=True)
        return summary_df
    
    def _format_results(self, name: str, eval_results) -> pd.DataFrame:
        """Format evaluation results into DataFrame."""
        if not eval_results:
            return pd.DataFrame({
                "Retriever Name": [name],
                "Hit Rate": [0.0],
                "MRR": [0.0]
            })

        metric_dicts = []
        for eval_result in eval_results:
            metric_dict = eval_result.metric_vals_dict
            metric_dicts.append(metric_dict)

        if not metric_dicts:
            return pd.DataFrame({
                "Retriever Name": [name],
                "Hit Rate": [0.0],
                "MRR": [0.0]
            })

        full_df = pd.DataFrame(metric_dicts)
        hit_rate = full_df.get("hit_rate", pd.Series([0.0])).mean()
        mrr = full_df.get("mrr", pd.Series([0.0])).mean()

        return pd.DataFrame({
            "Retriever Name": [name], 
            "Hit Rate": [hit_rate], 
            "MRR": [mrr]
        })
    
    def save_results(self, results_df: pd.DataFrame):
        """Save evaluation results to CSV and markdown."""
        # Save CSV
        csv_path = self.results_base_path / "retriever_evaluation_results.csv"
        results_df.to_csv(csv_path, index=False)
        logging.info(f"Saved CSV results to {csv_path}")
        
        # Generate and save markdown report
        md_report = self._generate_markdown_report(results_df)
        md_path = self.results_base_path / f"{self.run_name}.md"
        with open(md_path, "w") as f:
            f.write(md_report)
        logging.info(f"Saved markdown report to {md_path}")
        
        # Print summary to console
        print("\n" + "=" * 80)
        print("EVALUATION RESULTS SUMMARY")
        print("=" * 80)
        print(results_df.to_string(index=False))
        print("=" * 80)
        
        # Find best configuration
        best_idx = results_df["Hit Rate"].idxmax()
        best_config = results_df.iloc[best_idx]
        print(f"\nBest Configuration: {best_config['Retriever Name']}")
        print(f"  Hit Rate: {best_config['Hit Rate']:.2f}")
        print(f"  MRR: {best_config['MRR']:.3f}")
        print(f"\nResults saved to: {csv_path}")
        print(f"Report saved to: {md_path}")
    
    def _generate_markdown_report(self, results_df: pd.DataFrame) -> str:
        """Generate simplified markdown evaluation report."""
        report = f"""# {self.run_name}

## Description
{self.run_description}

## Configuration
- **Embedding Model**: {self.params['EMBEDDING_MODEL']}
- **Generation Model**: {self.params['GENERATION_MODEL']}

## Results Summary

| TOP_K | Hit Rate | MRR    |
|-------|----------|--------|
"""
        
        for _, row in results_df.iterrows():
            k_val = row['Retriever Name'].replace('top_', '')
            report += f"| {k_val:5} | {row['Hit Rate']:.4f}   | {row['MRR']:.6f} |\n"
        
        return report
    
    def run(self):
        """Execute the complete evaluation pipeline."""
        logging.info("=" * 80)
        logging.info(f"Starting Evaluation: {self.run_name}")
        logging.info("=" * 80)
        
        # Initialize agent
        logging.info("Initializing ParliamentAgent...")
        agent = ParliamentAgent()
        
        # Get all nodes
        logging.info("Retrieving nodes from index...")
        retriever = agent._index.as_retriever(similarity_top_k=1000)
        dummy_results = retriever.retrieve("parliamentary meeting scotland")
        nodes = [result.node for result in dummy_results]
        
        if len(nodes) == 0:
            logging.warning("Dummy query returned 0 nodes, using Qdrant scroll...")
            nodes = self._get_nodes_from_qdrant(agent)
        
        logging.info(f"Found {len(nodes)} nodes in the index")
        
        if len(nodes) == 0:
            logging.error("No nodes found! Cannot proceed with evaluation.")
            return
        
        # Apply max_nodes limit if specified
        if self.max_nodes and self.max_nodes < len(nodes):
            logging.info(f"Limiting to first {self.max_nodes} nodes (MAX_NODES_FOR_EVAL)")
            nodes = nodes[:self.max_nodes]
        
        # Generate or load questions
        if self.dataset_path.exists():
            logging.info("=" * 80)
            logging.info(f"DATASET ALREADY EXISTS - Skipping question generation")
            logging.info(f"Loading existing evaluation dataset from {self.dataset_path}")
            logging.info("=" * 80)
            eval_dataset = EmbeddingQAFinetuneDataset.from_json(str(self.dataset_path))
            logging.info(f"Loaded {len(eval_dataset.queries)} questions from existing dataset")
        else:
            logging.info("=" * 80)
            logging.info("DATASET NOT FOUND - Generating new questions...")
            logging.info("=" * 80)
            eval_dataset = self.generate_questions(nodes, Settings.llm)
            
            # Save dataset
            self.dataset_path.parent.mkdir(parents=True, exist_ok=True)
            eval_dataset.save_json(str(self.dataset_path))
            logging.info(f"Saved evaluation dataset to {self.dataset_path}")
        
        total_questions = len(eval_dataset.queries)
        if total_questions == 0:
            logging.error("No questions in evaluation dataset!")
            return
        
        # Evaluate retriever
        logging.info("Starting retriever evaluation...")
        results_df = self.evaluate_retriever(agent, eval_dataset)
        
        # Save results
        self.save_results(results_df)
        
        logging.info("=" * 80)
        logging.info("Evaluation completed successfully!")
        logging.info("=" * 80)
    
    def _get_nodes_from_qdrant(self, agent: ParliamentAgent) -> List[TextNode]:
        """Fallback: Get nodes directly from Qdrant."""
        client = agent._qdrant_manager.client
        collection_name = agent._qdrant_manager.collection_name
        
        collection_info = client.get_collection(collection_name)
        logging.info(f"Qdrant collection has {collection_info.points_count} points")
        
        if collection_info.points_count == 0:
            return []
        
        logging.info("Fetching all points from Qdrant...")
        all_points = []
        offset = None
        while True:
            points, next_offset = client.scroll(
                collection_name=collection_name,
                limit=100,
                offset=offset,
                with_payload=True,
                with_vectors=False
            )
            all_points.extend(points)
            if next_offset is None:
                break
            offset = next_offset
        
        logging.info(f"Retrieved {len(all_points)} points from Qdrant")
        
        nodes = []
        for point in all_points:
            node = TextNode(
                text=point.payload.get("text", ""),
                id_=str(point.id),
                metadata=point.payload
            )
            nodes.append(node)
        
        return nodes


if __name__ == "__main__":
    evaluator = ParliamentaryEvaluator()
    evaluator.run()
