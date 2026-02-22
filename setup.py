from setuptools import setup, find_packages

setup(
    name="parliament_agent",
    version="0.1.0",
    description="RAG-based conversational agent for Scottish Parliament documents",
    author="Your Name",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "llama-index",
        "llama-index-llms-gemini",
        "llama-index-embeddings-huggingface",
        "llama-index-vector-stores-qdrant",
        "qdrant-client",
        "sentence-transformers",
        "python-dotenv",
        "pyyaml",
        "fastapi",
        "uvicorn",
        "ragas",
    ],
)
