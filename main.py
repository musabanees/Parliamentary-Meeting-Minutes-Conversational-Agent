"""
FastAPI application for Parliamentary Meeting Minutes conversational agent.
"""
import logging
from contextlib import asynccontextmanager
from pathlib import Path
from datetime import datetime

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import uvicorn

from parliament_agent.agent import ParliamentAgent

# Configure logging to save to logs folder
LOGS_DIR = Path("logs")
LOGS_DIR.mkdir(exist_ok=True)
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
log_file = LOGS_DIR / f"api_{timestamp}.log"

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ],
    force=True
)
logger = logging.getLogger(__name__)
logger.info(f"API logging to {log_file}")

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

class ChatRequest(BaseModel):
    query: str

class ChatResponse(BaseModel):
    """Response model for the chat endpoint."""
    response: str
    content: Optional[List[str]] = None
    scores: Optional[List[float]] = None


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

        result = await agent.chat(query=request.query)

        return ChatResponse(
            response=result["answer"],
            content=result.get("content"),
            scores=result.get("scores"),
        )
    except Exception as e:
        logger.exception("Error processing chat request")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
