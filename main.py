"""
FastAPI application for Parliamentary Meeting Minutes conversational agent.
"""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import uvicorn

from parliament_agent.agent import ParliamentAgent

app = FastAPI(
    title="Parliamentary Minutes Chat API",
    description="A conversational AI agent for querying parliamentary meeting minutes",
    version="1.0.0",
)

# TODO: Initialise your ParliamentAgent here
# agent = ParliamentAgent()


class Message(BaseModel):
    """Represents a single message in the conversation."""
    role: str  # "user" or "assistant"
    content: str


class ChatRequest(BaseModel):
    """Request model for the chat endpoint."""
    query: str
    history: Optional[List[Message]] = []


class ChatResponse(BaseModel):
    """Response model for the chat endpoint."""
    response: str
    sources: Optional[List[str]] = None  # Optional: citations/sources


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Chat endpoint for conversational queries about parliamentary minutes.

    Args:
        request: ChatRequest containing the user query and conversation history

    Returns:
        ChatResponse with the agent's response and optional sources
    """
    try:
        # TODO: Implement the chat logic using your ParliamentAgent
        # Example usage:
        # response = agent.chat(query=request.query, history=request.history)
        # return ChatResponse(response=response["answer"], sources=response.get("sources"))

        raise NotImplementedError(
            "Chat endpoint not yet implemented. "
            "Please implement the ParliamentAgent and wire it up here."
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
