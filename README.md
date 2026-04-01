# Parliamentary Meeting Minutes Conversational Agent

**Time Estimate** : We encourage you to spend no more than half a day on this. We value quality and thoughtful design over a rushed, complex solution.

## Context

This repository contains the base code for a simple API service. We have a small but interesting dataset of parliamentary meeting minutes (in the `/data` folder), and our goal is to make this data accessible through a natural language conversation.

Your task is to contribute a new feature to this service: a conversational AI agent that allows a user to "chat" with these documents to find insights.

This isn't just a "find" or "search" task. We're looking for a system that can synthesise information, reason across documents, and provide grounded answers based only on the provided text.

## Your Task

Your goal is to implement the backend logic for this conversational agent. You will build this feature on a new git branch and submit it via a Pull Request (PR) for us to review.

Your implementation should aim to deliver the following:

1. **Implement a Context Retrieval Pipeline**: Consider a modern Retrieval-Augmented Generation (RAG) approach.

2. **Build a Conversational Agent**: Your agent should use an LLM (via API or local) to synthesise an answer from the retrieved context.

3. **Handle Chat History**: The agent must be able to handle follow-up questions (e.g., "What did he say next?", "Who disagreed with that point?"). The `/chat` endpoint skeleton is set up to receive this history.

4. **Ensure Groundedness**: The agent's primary challenge is to avoid hallucination. It should answer based only on the retrieved context and gracefully state when an answer isn't available in the documents.

5. **Wire up the API**: You'll find a skeleton `agent.py` and `main.py`. Your task is to build out the `ParliamentAgent` class and complete the `/chat` endpoint in `main.py` to use it.

## The Base Repository

This repo is a simple FastAPI application scaffold:

- **`/data/`**: Contains the sample meeting minutes
- **`/parliament_agent/agent.py`**: A skeleton file for your core agent logic. This is where most of your work will go
- **`/main.py`**: The FastAPI application. It has a working `/health` check and a skeleton `/chat` endpoint for you to complete
- **`/tests/`**: We've included a simple API test. We strongly encourage you to add tests for your new agent's logic
- **`Dockerfile`**: A base Dockerfile to get you started. Your final submission should be runnable with Docker (ideally via `docker-compose up`) but you may conisder other environment options if you have a preference.

## Requirements

1. **Context Retrieval** - Implement a context retrieval pipeline
2. **Conversational Memory** - Agent should handle multi-turn conversations with context
3. **Grounded Responses** - Avoid hallucination; answer only from retrieved context
4. **Working API** - Complete the `/chat` endpoint to accept queries and return responses
5. **Reproducibility** - Ideally run with `docker-compose up`

## Acceptance Criteria

- [ ] Context retrieval pipeline
- [ ] `/chat` endpoint accepts user queries and conversation history
- [ ] Working Conversational Agent 
- [ ] Clear explanation of your approach in PR description

## Nice-to-have

- [ ] Evaluation metrics for retrieval quality or response groundedness
- [ ] Comparison of different chunking strategies or embedding models
- [ ] Observability/logging for debugging retrieval and generation
- [ ] Citation support (referencing which document/section an answer came from)

## Submission Guidelines

1. Create a branch from `main`: `feature/conversational-agent` (or similar descriptive name)
2. Make your changes
3. Submit a Pull Request with:
   - Clear description of what you implemented
   - Explanation of design decisions (choice of embedding model, vector store, LLM, chunking strategy, etc.)
   - Any trade-offs or limitations you identified
   - Instructions for testing your implementation
4. Be ready to respond to PR review comments

## Getting Started

### Option 1: Docker (Recommended)

```bash
# You'll need to create a docker-compose.yml file
# Then build and run the service
docker-compose up

# The API should be available at http://localhost:8000
# View docs at http://localhost:8000/docs
```

### Option 2: Local Development with Virtual Environment

```bash
# Setup virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run tests
pytest tests/ -v

# Run the application
python main.py
```

## What We're Evaluating

- **LLM/Agentic Thinking** - Understanding of modern RAG architectures and conversational AI
- **Engineering Competency** - Clean code, proper abstractions, error handling
- **Problem Framing** - How you approach the open-ended challenge of grounding LLM responses
- **Design Decisions** - Thoughtful choices around chunking, embedding, retrieval and prompting
- **Testing Mindset** - Appropriate tests for your agent's behaviour
- **Communication** - Clear explanation of approach, trade-offs and limitations

## Notes

- Use any tools or frameworks you feel are suitable for the task, explain why you chose them for your solution
- You're free to use any LLM provider (OpenAI, Anthropic, local models, etc.)
- You may use any vector store or embedding model you prefer (a numpy array is an acceptable vector store!)
- Focus on demonstrating understanding of RAG principles and grounding strategies
- Simple, clear implementations are better than over-engineered solutions
- If you use API keys, include clear instructions on how to set them (`.env.example` file recommended)
- You don't need production-level scalability, but your design should show awareness of potential issues
- If you can see a better approach to deliver the feature and capability, you do not need to follow the exact approach outlined - just explain clearly why you made certain decisions in your PR.

**Good Luck!**
We look forward to reviewing your solution. If you get stuck or want to clarify something, please reach out.

