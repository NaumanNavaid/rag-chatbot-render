"""
FastAPI RAG Chatbot Backend using OpenAI
=========================================
RAG-based AI tutor for Physical AI & Humanoid Robotics textbook
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import asyncio
import logging
import os
import random
import time

from dotenv import load_dotenv
from agents import Agent, Runner, OpenAIChatCompletionsModel
from agents import set_tracing_disabled, function_tool
from openai import AsyncOpenAI, APIError

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

# ---------------------------------------------------
# Logging Setup
# ---------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------
# Environment Configuration
# ---------------------------------------------------
load_dotenv()
set_tracing_disabled(disabled=True)

# ---------------------------------------------------
# FastAPI App Configuration
# ---------------------------------------------------
app = FastAPI(
    title="AI Tutor API",
    description="RAG-based AI tutor for Physical AI & Humanoid Robotics textbook",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your frontend domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------
# Model and Client Initialization
# ---------------------------------------------------
# OpenAI Configuration
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    raise ValueError("Missing OPENAI_API_KEY in environment variables")

# Initialize OpenAI clients
chat_client = AsyncOpenAI(api_key=openai_api_key)
embedding_client = AsyncOpenAI(api_key=openai_api_key)

# Chat Model
model = OpenAIChatCompletionsModel(
    model="gpt-4o-mini",  # Using gpt-4o-mini for cost efficiency
    openai_client=chat_client
)

# Qdrant Configuration
qdrant_url = os.getenv("QDRANT_URL")
qdrant_api_key = os.getenv("QDRANT_API_KEY")
collection_name = os.getenv("COLLECTION_NAME", "humanoid_ai_book_openai")
embed_model = os.getenv("EMBED_MODEL", "text-embedding-3-small")

if not qdrant_url or not qdrant_api_key:
    raise ValueError("Missing QDRANT_URL or QDRANT_API_KEY in environment variables")

qdrant = QdrantClient(
    url=qdrant_url,
    api_key=qdrant_api_key
)

# ---------------------------------------------------
# Pydantic Schemas
# ---------------------------------------------------
class ChatRequest(BaseModel):
    """Request schema for chat endpoint"""
    message: str
    conversation_id: Optional[str] = None
    session_id: Optional[str] = None
    stream: bool = False  # For future streaming support

class Source(BaseModel):
    """Schema for a source document"""
    text: str
    url: Optional[str] = None
    score: Optional[float] = None

class ChatResponse(BaseModel):
    """Response schema for chat endpoint"""
    response: str
    sources: List[Source] = []
    conversation_id: Optional[str] = None
    session_id: Optional[str] = None
    model_used: str
    retrieval_count: int

class HealthResponse(BaseModel):
    """Response schema for health check"""
    status: str
    message: str
    details: Optional[Dict[str, Any]] = None

# ---------------------------------------------------
# Embedding and Retrieval Functions
# ---------------------------------------------------
async def get_embedding(text: str) -> List[float]:
    """Generate embedding for the given text using OpenAI"""
    try:
        response = await embedding_client.embeddings.create(
            model=embed_model,
            input=text
        )
        return response.data[0].embedding
    except APIError as e:
        logger.error(f"OpenAI embedding error: {e}")
        raise HTTPException(status_code=500, detail="Failed to generate embedding")
    except Exception as e:
        logger.error(f"Unexpected embedding error: {e}")
        raise HTTPException(status_code=500, detail="Embedding service unavailable")

@function_tool
async def retrieve(query: str) -> List[str]:
    """
    Retrieve relevant textbook excerpts from Qdrant database
    based on the user's query.
    """
    try:
        # Generate embedding for the query
        embedding = await get_embedding(query)

        # Search Qdrant
        search_result = qdrant.query_points(
            collection_name=collection_name,
            query=embedding,
            limit=5,
            with_payload=True,
            with_vectors=False
        )

        # Extract text from results
        retrieved_texts = []
        for point in search_result.points:
            if hasattr(point, 'payload') and point.payload:
                text = point.payload.get('text', '')
                if text:
                    retrieved_texts.append(text)

        logger.info(f"Retrieved {len(retrieved_texts)} documents for query: {query[:50]}...")
        return retrieved_texts

    except Exception as e:
        logger.error(f"Retrieval error: {e}")
        return []  # Return empty list to allow the agent to continue

# ---------------------------------------------------
# AI Agent Configuration
# ---------------------------------------------------
agent = Agent(
    name="AI Tutor",
    instructions="""
You are an expert AI tutor specializing in Physical AI and Humanoid Robotics.

Your guidelines:
1. Always use the retrieve tool first to get relevant information from the textbook
2. Answer questions based ONLY on the retrieved content
3. If the information isn't in the retrieved content, say "I don't have enough information to answer that question"
4. Provide clear, educational explanations
5. Be concise but thorough in your answers
6. Cite the relevant information from your sources

Important: Never make up information. If you're not sure based on the retrieved content, admit it.
""",
    model=model,
    tools=[retrieve]
)

# ---------------------------------------------------
# Retry Logic for API Calls
# ---------------------------------------------------
async def run_with_retry(agent: Agent, user_input: str, max_retries: int = 3) -> Any:
    """
    Execute the agent with retry logic for handling rate limits
    """
    base_delay = 1.0
    max_delay = 10.0

    for attempt in range(max_retries):
        try:
            result = await Runner.run(agent, input=user_input)
            return result

        except APIError as e:
            # Check if it's a rate limit error
            if "rate_limit" in str(e).lower() or "429" in str(e):
                if attempt < max_retries - 1:
                    # Exponential backoff with jitter
                    delay = min(base_delay * (2 ** attempt), max_delay)
                    jitter = random.uniform(0.5, 1.5)
                    wait_time = delay * jitter

                    logger.warning(
                        f"Rate limit hit (attempt {attempt + 1}/{max_retries}). "
                        f"Waiting {wait_time:.2f}s before retry..."
                    )
                    await asyncio.sleep(wait_time)
                    continue
                else:
                    raise HTTPException(
                        status_code=429,
                        detail="Service temporarily unavailable due to rate limits. Please try again later."
                    )
            else:
                # Different API error
                logger.error(f"OpenAI API error: {e}")
                raise HTTPException(status_code=500, detail="AI service error")

        except Exception as e:
            logger.error(f"Unexpected error in agent execution: {e}")
            if attempt < max_retries - 1:
                await asyncio.sleep(base_delay)
                continue
            raise HTTPException(status_code=500, detail="Internal server error")

    raise HTTPException(status_code=500, detail="Failed after multiple retries")

# ---------------------------------------------------
# API Endpoints
# ---------------------------------------------------
@app.get("/", response_model=HealthResponse)
async def root():
    """Root endpoint with basic info"""
    return HealthResponse(
        status="healthy",
        message="AI Tutor API is running",
        details={
            "model": model.model,
            "embedding_model": embed_model,
            "collection": collection_name
        }
    )

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Comprehensive health check"""
    health_status = {
        "openai": False,
        "qdrant": False,
        "embedding_model": embed_model,
        "chat_model": model.model
    }

    try:
        # Test OpenAI embedding
        test_embedding = await get_embedding("test")
        if test_embedding:
            health_status["openai"] = True
    except Exception as e:
        logger.error(f"OpenAI health check failed: {e}")

    try:
        # Test Qdrant connection
        collections = qdrant.get_collections()
        collection_exists = any(
            c.name == collection_name
            for c in collections.collections
        )
        if collection_exists:
            collection_info = qdrant.get_collection(collection_name)
            health_status["qdrant"] = True
            health_status["collection_points"] = collection_info.points_count
            health_status["vector_size"] = collection_info.config.params.vectors.size
    except Exception as e:
        logger.error(f"Qdrant health check failed: {e}")

    all_healthy = all([health_status["openai"], health_status["qdrant"]])

    return HealthResponse(
        status="healthy" if all_healthy else "degraded",
        message="All services operational" if all_healthy else "Some services unavailable",
        details=health_status
    )

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Main chat endpoint for interacting with the AI tutor
    """
    start_time = time.time()

    try:
        logger.info(f"Received message: {request.message[:100]}...")

        # Run the agent with retry logic
        result = await run_with_retry(agent, request.message)

        # Extract sources from the agent's tool calls
        sources = []
        retrieval_count = 0

        if hasattr(result, 'raw_responses') and result.raw_responses:
            for response in result.raw_responses:
                if hasattr(response, 'tool_calls'):
                    for tool_call in response.tool_calls:
                        if tool_call.function.name == "retrieve":
                            # Get the retrieved documents
                            retrieved_texts = tool_call.function.result
                            if isinstance(retrieved_texts, list):
                                retrieval_count = len(retrieved_texts)
                                # Create Source objects (we don't have URLs/scores from the tool)
                                sources = [
                                    Source(text=text[:500] + "..." if len(text) > 500 else text)
                                    for text in retrieved_texts
                                ]

        # Calculate response time
        response_time = time.time() - start_time
        logger.info(f"Response generated in {response_time:.2f}s with {retrieval_count} sources")

        return ChatResponse(
            response=result.final_output,
            sources=sources,
            conversation_id=request.conversation_id,
            session_id=request.session_id,
            model_used=model.model,
            retrieval_count=retrieval_count
        )

    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"Chat endpoint error: {e}")
        raise HTTPException(status_code=500, detail="Failed to process chat request")

# ---------------------------------------------------
# Additional Utility Endpoints
# ---------------------------------------------------
@app.get("/collection/stats")
async def get_collection_stats():
    """Get statistics about the Qdrant collection"""
    try:
        collection_info = qdrant.get_collection(collection_name)
        return {
            "collection_name": collection_name,
            "points_count": collection_info.points_count,
            "vector_size": collection_info.config.params.vectors.size,
            "distance_metric": collection_info.config.params.vectors.distance
        }
    except Exception as e:
        logger.error(f"Error getting collection stats: {e}")
        raise HTTPException(status_code=404, detail="Collection not found or error")

@app.delete("/collection")
async def clear_collection():
    """Clear all points from the collection (use with caution!)"""
    try:
        qdrant.delete_collection(collection_name)
        # Recreate collection
        vector_size = 1536 if embed_model == "text-embedding-3-small" else 3072
        qdrant.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE)
        )
        return {"message": f"Collection '{collection_name}' cleared and recreated"}
    except Exception as e:
        logger.error(f"Error clearing collection: {e}")
        raise HTTPException(status_code=500, detail="Failed to clear collection")

# ---------------------------------------------------
# Run Server (Development)
# ---------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(description="AI Tutor API Server")
    parser.add_argument("--port", type=int, default=int(os.getenv("PORT", 8000)),
                        help="Port to run the server on")
    args = parser.parse_args()

    # Configure logging for uvicorn
    if args.port == int(os.getenv("PORT", 8000)):
        # Default port, use reload
        uvicorn_config = {
            "app": "main:app",
            "host": "0.0.0.0",
            "port": args.port,
            "reload": True,
            "log_level": "info"
        }
    else:
        # Custom port, don't use reload
        uvicorn_config = {
            "app": app,
            "host": "0.0.0.0",
            "port": args.port,
            "reload": False,
            "log_level": "info"
        }

    logger.info(f"Starting AI Tutor API server on port {args.port}...")
    uvicorn.run(**uvicorn_config)