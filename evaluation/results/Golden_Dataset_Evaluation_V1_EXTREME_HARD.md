# Golden Dataset Evaluation Results (RAGAS + LlamaIndex)
## Description
Evaluation of Parliamentary RAG system using human-curated golden dataset with RAGAS metrics via LlamaIndex integration.

## Configuration
- **Embedding Model**: intfloat/e5-small-v2
- **Generation Model**: gpt-4o-mini
- **Evaluation Framework**: RAGAS (LlamaIndex Integration)
- **Judge LLM**: gpt-4.1-mini (OpenAI)
- **Total Examples**: 10

## RAGAS Metrics Summary

- **faithfulness**: 0.7290
- **context_precision**: 0.7517
- **context_recall**: 0.7400
