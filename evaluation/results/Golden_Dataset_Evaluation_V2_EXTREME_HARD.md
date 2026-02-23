# Golden Dataset Evaluation Results (RAGAS + LlamaIndex)
## Description
Evaluation of Parliamentary RAG system using human-curated golden dataset with RAGAS metrics via LlamaIndex integration.

## Configuration
- **Embedding Model**: text-embedding-3-small
- **Generation Model**: gpt-4o-mini
- **Evaluation Framework**: RAGAS (LlamaIndex Integration)
- **Judge LLM**: gpt-4.1-mini (OpenAI)
- **Total Examples**: 10

## RAGAS Metrics Summary

- **faithfulness**: 0.8639
- **context_precision**: 0.9800
- **context_recall**: 0.7050
