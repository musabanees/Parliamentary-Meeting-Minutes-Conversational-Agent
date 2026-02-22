import json
import logging
from pathlib import Path
from datetime import datetime
import os

import yaml
import pandas as pd
from tqdm import tqdm
from dotenv import load_dotenv

from ragas.metrics import (
    Faithfulness,
    ContextPrecision,
    ContextRecall,
)
from ragas.llms import llm_factory
from ragas.integrations.llama_index import evaluate
from ragas import EvaluationDataset, SingleTurnSample

from openai import OpenAI as OpenAIClient

from parliament_agent.agent import ParliamentAgent


class GoldenDatasetEvaluator:
    """Evaluates RAG system using golden dataset with RAGAS metrics via LlamaIndex integration."""

    def __init__(self, params_path: str = "params.yaml"):
        """Initialize evaluator with configuration."""
        # Load environment variables
        load_dotenv()

        # Load params
        with open(params_path, "r") as f:
            self.params = yaml.safe_load(f)

        # Use paths from params.yaml
        self.golden_dataset_path = Path(self.params["GOLDEN_DATASET_PATH"])
        self.results_path = Path(self.params["EVALUATION_RESULTS_PATH"])
        self.results_path.mkdir(parents=True, exist_ok=True)

        # Setup logging
        self._setup_logging()

        # Verify OpenAI API key
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        if not self.openai_api_key:
            raise ValueError("OPENAI_API_KEY not found in environment variables")

        os.environ["OPENAI_API_KEY"] = self.openai_api_key

        logging.info("Initialized GoldenDatasetEvaluator with RAGAS + LlamaIndex")
        logging.info(f"Using embedding model: {self.params['EMBEDDING_MODEL']}")
        logging.info(f"Using generation model: {self.params['GENERATION_MODEL']}")
        logging.info(f"Using judge model: {self.params['JUDGE_MODEL']}")
        logging.info(f"Golden dataset path: {self.golden_dataset_path}")
        logging.info(f"Results path: {self.results_path}")

    def _setup_logging(self):
        """Setup logging to both console and file."""
        logs_dir = Path("logs")
        logs_dir.mkdir(exist_ok=True)

        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        log_file = logs_dir / f"golden_evaluation_{timestamp}.log"

        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ],
            force=True
        )
        logging.info(f"Logging to {log_file}")

    def load_golden_dataset(self):
        """Load golden dataset from JSON."""
        logging.info(f"Loading golden dataset from {self.golden_dataset_path}")

        with open(self.golden_dataset_path, "r") as f:
            data = json.load(f)

        logging.info(f"Found {len(data)} examples in golden dataset")
        return data

    def prepare_ragas_dataset(self, golden_data: list):
        """
        Prepare dataset in RAGAS EvaluationDataset format.

        RAGAS expects SingleTurnSample objects with:
        - user_input: The query
        - reference: Reference answer (for context_recall)
        - reference_contexts: Reference contexts (for context_precision)
        """
        logging.info("Preparing RAGAS EvaluationDataset...")

        samples = []

        for idx, item in enumerate(golden_data):
            query = item["query"]
            reference_answer = item["reference_answer"]
            reference_contexts = item["reference_contexts"]

            logging.info(f"Processing query {idx + 1}/{len(golden_data)}: {query[:100]}...")

            try:
                # Create SingleTurnSample
                sample = SingleTurnSample(
                    user_input=query,
                    reference=reference_answer,
                    reference_contexts=reference_contexts
                )
                samples.append(sample)

            except Exception as e:
                logging.error(f"Error processing query {idx + 1}: {e}")
                continue

        logging.info(f"Prepared {len(samples)} samples for RAGAS evaluation")

        # Convert to EvaluationDataset
        dataset = EvaluationDataset(samples=samples)
        return dataset

    def run_evaluation(self):
        """Execute the complete golden dataset evaluation pipeline."""
        logging.info("=" * 80)
        logging.info("Starting Golden Dataset Evaluation with RAGAS + LlamaIndex")
        logging.info("=" * 80)

        # Load golden dataset
        golden_data = self.load_golden_dataset()

        # Initialize ParliamentAgent
        logging.info("Initializing ParliamentAgent...")
        agent = ParliamentAgent()

        # Get query engine from agent's index
        logging.info("Creating query engine from agent index...")
        query_engine = agent._index.as_query_engine(similarity_top_k=4, response_mode="compact")

        # Prepare RAGAS dataset
        ragas_dataset = self.prepare_ragas_dataset(golden_data)

        # Configure RAGAS with models from params.yaml
        logging.info("Configuring RAGAS evaluation metrics...")

        # Judge LLM from params.yaml
        judge_model = self.params["JUDGE_MODEL"]
        logging.info(f"Using judge LLM: {judge_model}")

        # Create OpenAI client and use llm_factory
        openai_client = OpenAIClient(api_key=self.openai_api_key)
        evaluator_llm = llm_factory(
            model=judge_model,
            client=openai_client,
            temperature=0,
        )

        # Initialize metrics with the LLM
        metrics = [
            Faithfulness(llm=evaluator_llm),
            ContextPrecision(llm=evaluator_llm),
            ContextRecall(llm=evaluator_llm),
        ]

        # Run RAGAS evaluation with LlamaIndex integration
        logging.info("Running RAGAS evaluation with LlamaIndex integration...")
        logging.info("Metrics:")
        logging.info("  - faithfulness (hallucination detection)")
        logging.info("  - context_precision (retrieval quality)")
        logging.info("  - context_recall (retrieval completeness)")
        logging.info("This may take several minutes...")

        try:
            result = evaluate(
                query_engine=query_engine,
                metrics=metrics,
                dataset=ragas_dataset,
            )

            # Convert to DataFrame
            results_df = result.to_pandas()

            # Save results
            self._save_results(results_df, result)

            # Display summary
            self._display_summary(result, results_df)

            logging.info("=" * 80)
            logging.info("Golden Dataset Evaluation completed successfully!")
            logging.info("=" * 80)

            return result

        except Exception as e:
            logging.error(f"Evaluation failed: {e}", exc_info=True)
            raise

    def _save_results(self, results_df: pd.DataFrame, result):
        """Save evaluation results to markdown report."""
        # Generate and save markdown summary
        md_report = self._generate_markdown_report(result, results_df)
        md_path = self.results_path / "Golden_Dataset_Evaluation.md"
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(md_report)
        logging.info(f"Saved markdown report to {md_path}")

    def _generate_markdown_report(self, result, results_df: pd.DataFrame) -> str:
        """Generate markdown report from RAGAS results."""
        # Convert result to dict format for easier access
        # result is an EvaluationResult object with scores as attributes

        report = f"""# Golden Dataset Evaluation Results (RAGAS + LlamaIndex)
                    ## Description
                    Evaluation of Parliamentary RAG system using human-curated golden dataset with RAGAS metrics via LlamaIndex integration.

                    ## Configuration
                    - **Embedding Model**: {self.params['EMBEDDING_MODEL']}
                    - **Generation Model**: {self.params['GENERATION_MODEL']}
                    - **Evaluation Framework**: RAGAS (LlamaIndex Integration)
                    - **Judge LLM**: {self.params['JUDGE_MODEL']} (OpenAI)
                    - **Total Examples**: {len(results_df)}

                    ## RAGAS Metrics Summary

                    """

        # Add metric scores from the dataframe
        # Calculate mean scores for each metric column
        metric_columns = [col for col in results_df.columns if col not in ['user_input', 'retrieved_contexts', 'reference_contexts', 'response', 'reference']]

        for metric_name in metric_columns:
            if metric_name in results_df.columns:
                mean_score = results_df[metric_name].mean()
                report += f"- **{metric_name}**: {mean_score:.4f}\n"

        return report

    def _display_summary(self, result, results_df: pd.DataFrame):
        """Display evaluation summary to console."""
        print("\n" + "=" * 80)
        print("RAGAS EVALUATION SUMMARY (LlamaIndex Integration)")
        print("=" * 80)

        print(f"Total Examples: {len(results_df)}")
        print("\nMetric Scores:")

        # Calculate mean scores from dataframe
        metric_columns = [col for col in results_df.columns if col not in ['user_input', 'retrieved_contexts', 'reference_contexts', 'response', 'reference']]

        for metric_name in metric_columns:
            if metric_name in results_df.columns:
                mean_score = results_df[metric_name].mean()
                print(f"  {metric_name}: {mean_score:.4f}")


if __name__ == "__main__":
    evaluator = GoldenDatasetEvaluator()
    # Run evaluation on all samples
    evaluator.run_evaluation()
