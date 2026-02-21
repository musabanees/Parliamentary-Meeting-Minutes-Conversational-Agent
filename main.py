"""
FastAPI application for Parliamentary Meeting Minutes conversational agent.
"""
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import uvicorn

from parliament_agent.agent import ParliamentAgent

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

agent: Optional[ParliamentAgent] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global agent
    logger.info("Starting up – initialising ParliamentAgent...")
    agent = ParliamentAgent()
    logger.info("ParliamentAgent ready")
    yield
    logger.info("Shutting down")


app = FastAPI(
    title="Parliamentary Minutes Chat API",
    description="A conversational AI agent for querying parliamentary meeting minutes",
    version="1.0.0",
    lifespan=lifespan,
)


class Message(BaseModel):
    """Represents a single message in the conversation."""
    role: str
    content: str


class ChatRequest(BaseModel):
    """Request model for the chat endpoint."""
    query: str
    history: Optional[List[Message]] = []


class ChatResponse(BaseModel):
    """Response model for the chat endpoint."""
    response: str
    sources: Optional[List[str]] = None


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Chat endpoint for conversational queries about parliamentary minutes.

    Accepts a user query and optional conversation history,
    returns a grounded response with source citations.
    """
    if agent is None:
        raise HTTPException(status_code=503, detail="Agent not initialised yet")

    try:
        history_dicts = [
            {"role": msg.role, "content": msg.content}
            for msg in (request.history or [])
        ]

        result = await agent.chat(query=request.query, history=history_dicts)

        return ChatResponse(
            response=result["answer"],
            sources=result.get("sources"),
        )
    except Exception as e:
        logger.exception("Error processing chat request")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
