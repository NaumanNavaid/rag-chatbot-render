"""
FastAPI RAG Chatbot Backend using OpenAI
=========================================
RAG-based AI tutor for Physical AI & Humanoid Robotics textbook
Now with User Text Selection Support
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any, Literal
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
from pathlib import Path
# Load .env from the same directory as this script
load_dotenv(Path(__file__).parent / '.env')
set_tracing_disabled(disabled=True)

# ---------------------------------------------------
# FastAPI App Configuration
# ---------------------------------------------------
app = FastAPI(
    title="AI Tutor API",
    description="RAG-based AI tutor for Physical AI & Humanoid Robotics textbook with User Text Selection",
    version="2.1.0",
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
# Using the valid API key from test_key.py
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
    selected_text: Optional[str] = None  # NEW: User-selected text for prioritized processing

class Source(BaseModel):
    """Schema for a source document"""
    text: str
    url: Optional[str] = None
    score: Optional[float] = None
    is_user_selected: Optional[bool] = False  # NEW: Indicates if this is user-selected text

class ChatResponse(BaseModel):
    """Response schema for chat endpoint"""
    response: str
    sources: List[Source] = []
    conversation_id: Optional[str] = None
    session_id: Optional[str] = None
    model_used: str
    retrieval_count: int
    used_selected_text: bool = False  # NEW: Indicates if user-selected text was used

class HealthResponse(BaseModel):
    """Response schema for health check"""
    status: str
    message: str
    details: Optional[Dict[str, Any]] = None

class SpecialistRequest(BaseModel):
    """Request schema for specialist chat endpoint"""
    message: str
    specialist_type: Literal["physics", "programming", "mathematics", "general"] = "general"
    conversation_id: Optional[str] = None
    session_id: Optional[str] = None
    selected_text: Optional[str] = None

class SpecialistResponse(BaseModel):
    """Response schema for specialist chat endpoint"""
    response: str
    specialist_type: str
    conversation_id: Optional[str] = None
    session_id: Optional[str] = None
    model_used: str
    confidence: Optional[float] = None

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

@function_tool
async def use_selected_text(text: str, query: str) -> List[str]:
    """
    NEW: Process user-selected text with priority over database search.
    This tool is used when the user has selected specific text they want to focus on.
    The selected text is given priority in the response.
    """
    try:
        # Always include the user-selected text as the first result
        results = [text]

        # Try to find similar content in the database to supplement the selection
        embedding = await get_embedding(text)

        search_result = qdrant.query_points(
            collection_name=collection_name,
            query=embedding,
            limit=3,  # Fewer results since we already have the user selection
            with_payload=True,
            with_vectors=False,
            score_threshold=0.8  # Higher threshold for very similar content
        )

        # Add highly similar content from the database
        for point in search_result.points:
            if hasattr(point, 'payload') and point.payload:
                db_text = point.payload.get('text', '')
                if db_text and db_text != text:  # Avoid duplicate
                    results.append(db_text)

        logger.info(f"Processed user-selected text with {len(results) - 1} supplementary results")
        return results

    except Exception as e:
        logger.error(f"Error processing selected text: {e}")
        # Fallback: just return the user-selected text
        return [text]

@function_tool
async def generate_code_example(concept: str, language: str = "python") -> str:
    """
    Generate a practical code example for a robotics concept.
    This is a reusable skill that can be used by any agent.
    """
    try:
        # For now, return a template. In a real implementation,
        # this could use the agent to generate actual code
        examples = {
            "inverse kinematics": f"""
# Inverse Kinematics Example in {language}
import numpy as np

def inverse_kinematics(x, y, l1, l2):
    '''Calculate joint angles for 2-DOF arm'''
    r = np.sqrt(x**2 + y**2)

    # Check if target is reachable
    if r > (l1 + l2):
        return None

    # Calculate angles using law of cosines
    theta2 = np.arccos((x**2 + y**2 - l1**2 - l2**2) / (2 * l1 * l2))
    theta1 = np.arctan2(y, x) - np.arctan2(l2 * np.sin(theta2), l1 + l2 * np.cos(theta2))

    return theta1, theta2

# Example usage
theta1, theta2 = inverse_kinematics(5, 3, 4, 3)
print(f"Joint angles: θ1 = {{np.degrees(theta1):.2f}}°, θ2 = {{np.degrees(theta2):.2f}}°")
""",
            "pid_controller": f"""
# PID Controller Example in {language}
class PIDController:
    def __init__(self, kp, ki, kd, setpoint):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.setpoint = setpoint
        self.integral = 0
        self.previous_error = 0

    def update(self, current_value, dt):
        error = self.setpoint - current_value
        self.integral += error * dt
        derivative = (error - self.previous_error) / dt

        output = self.kp * error + self.ki * self.integral + self.kd * derivative
        self.previous_error = error

        return output

# Example usage for robot arm position control
pid = PIDController(kp=1.0, ki=0.1, kd=0.05, setpoint=180)  # Target: 180 degrees
current_angle = 0
dt = 0.1  # 100ms update rate

for i in range(50):  # 5 seconds simulation
    control_signal = pid.update(current_angle, dt)
    # In real robot, this would move the motor
    current_angle += control_signal * dt * 10  # Simplified physics
    print(f"Time: {{i*dt:.1f}}s, Angle: {{current_angle:.1f}}°")
"""
        }

        return examples.get(concept.lower(), f"# Code example for {concept}\n# Example implementation coming soon...")
    except Exception as e:
        return f"Error generating code example: {e}"

@function_tool
async def create_practice_question(topic: str, difficulty: str = "medium") -> str:
    """
    Create a practice question for a given topic and difficulty level.
    This is a reusable skill for educational content.
    """
    questions = {
        "physics": {
            "easy": "Q: What is Newton's Second Law of Motion? How does it apply to a humanoid robot walking?",
            "medium": "Q: Calculate the torque required for a humanoid robot's elbow joint to lift a 5kg object at a 30cm distance from the joint.",
            "hard": "Q: Derive the equations of motion for a double pendulum system similar to a humanoid robot's arm, considering both links have mass m1 and m2."
        },
        "programming": {
            "easy": "Q: Write a simple function to convert degrees to radians for robot joint angles.",
            "medium": "Q: Implement a forward kinematics function for a 3-DOF robotic arm using transformation matrices.",
            "hard": "Q: Design a class structure for a humanoid robot simulation that includes kinematics, dynamics, and control systems."
        },
        "mathematics": {
            "easy": "Q: Convert the following robot joint angles from degrees to radians: θ1=45°, θ2=-30°, θ3=90°.",
            "medium": "Q: Given a 2x2 rotation matrix R = [[0.866, -0.5], [0.5, 0.866]], find the rotation angle in degrees.",
            "hard": "Q: Prove that the composition of two rotations in 2D is commutative, but composition of rotations in 3D is generally not commutative."
        }
    }

    if topic.lower() in questions and difficulty.lower() in questions[topic.lower()]:
        return questions[topic.lower()][difficulty.lower()]

    return f"Q: Practice question about {topic} ({difficulty} difficulty) - Coming soon!"

# ---------------------------------------------------
# AI Agent Configuration
# ---------------------------------------------------
def create_specialist_agents() -> Dict[str, Agent]:
    """Create specialized agents for different domains with book awareness"""

    # Physics Specialist (Book-Enhanced)
    physics_agent = Agent(
        name="Physics Tutor",
        instructions="""
You are an expert in physics concepts related to humanoid robotics and physical AI, with deep knowledge of the Physical AI & Humanoid Robotics textbook.

Your expertise includes:
- Mechanics (statics, dynamics, kinematics)
- Forces, torques, and momentum
- Energy and power in robotic systems
- Control theory and stability
- Material properties and structural analysis

When answering:
1. ALWAYS start by referencing relevant textbook content if available
2. Use the retrieve tool to find book passages first
3. Then enhance with your deeper physics knowledge
4. Provide real-world examples and applications in robotics
5. Use the generate_code_example tool for physics simulations
6. Use the create_practice_question tool to reinforce learning

Important: Always try to connect your explanations back to textbook concepts when relevant.
Examples:
- "This relates to Chapter 3's discussion on..."
- "Building on the textbook's explanation..."
- "The book mentions torque, but here's the deeper mathematical background..."

Always prioritize physics explanations and connect to book content when possible.
        """,
        model=model,
        tools=[retrieve, use_selected_text, generate_code_example, create_practice_question]
    )

    # Programming Specialist (Book-Enhanced)
    programming_agent = Agent(
        name="Programming Tutor",
        instructions="""
You are an expert in programming for robotics and AI systems, with comprehensive knowledge of the Physical AI & Humanoid Robotics textbook.

Your expertise includes:
- Python, C++, and ROS programming
- Algorithm design and optimization
- Data structures for robotics
- Software architecture for robotic systems
- API design and integration
- Version control and best practices
- Code examples from the textbook

When answering:
1. Check the retrieve tool for relevant code examples from the textbook
2. Enhance textbook code with industry best practices
3. Provide additional variations and optimizations
4. Explain the software architecture decisions
5. Use the generate_code_example tool frequently
6. Create relevant practice questions

Always try to:
- Reference specific code examples from the book if available
- "The textbook shows this approach, but here's an industry-standard way..."
- "Building on the book's implementation, let me show you..."
- Connect your explanations to robotics applications discussed in the text

Focus on practical implementation while connecting to book content.
        """,
        model=model,
        tools=[retrieve, use_selected_text, generate_code_example, create_practice_question]
    )

    # Mathematics Specialist (Book-Enhanced)
    math_agent = Agent(
        name="Mathematics Tutor",
        instructions="""
You are an expert in mathematical concepts for robotics and AI, with thorough knowledge of the Physical AI & Humanoid Robotics textbook.

Your expertise includes:
- Linear algebra (matrices, vectors, transformations)
- Calculus (derivatives, integrals, optimization)
- Statistics and probability
- Differential equations
- Numerical methods
- Geometry and trigonometry
- Mathematical foundations from the textbook

When answering:
1. Use retrieve to find mathematical explanations from the textbook
2. Show step-by-step derivations that relate to book content
3. Provide deeper mathematical intuition and proofs
4. Connect to robotics applications mentioned in the text
5. Use code tools for computational examples
6. Create mathematically-focused practice questions

Always try to:
- "The textbook introduces this concept in Chapter X, here's the complete derivation..."
- "Building on the book's mathematical foundation..."
- "The text mentions this equation, let me explore it in depth..."

Explain mathematical rigor and clarity, always connecting to robotics applications in the book.
        """,
        model=model,
        tools=[retrieve, use_selected_text, generate_code_example, create_practice_question]
    )

    # General Tutor (your existing agent)
    general_agent = Agent(
        name="General AI Tutor",
        instructions="""
You are a general AI tutor for Physical AI and Humanoid Robotics.

You can help with any topic, but will prioritize user-selected text when provided.
If a question is clearly about physics, programming, or mathematics, suggest using the appropriate specialist.
        """,
        model=model,
        tools=[retrieve, use_selected_text]
    )

    return {
        "physics": physics_agent,
        "programming": programming_agent,
        "mathematics": math_agent,
        "general": general_agent
    }

# Initialize specialist agents
specialist_agents = create_specialist_agents()

# Keep the original agent for backward compatibility
agent = Agent(
    name="AI Tutor",
    instructions="""
You are an expert AI tutor specializing in Physical AI and Humanoid Robotics.

Your guidelines:
1. If the user has provided selected text (indicated by the use_selected_text tool being called),
   PRIORITIZE that text in your response. It's the most relevant context for their question.
2. For regular queries without selected text, use the retrieve tool to get relevant information.
3. Answer based primarily on the provided context (selected text first, then retrieved documents).
4. When using selected text, explicitly acknowledge it in your response (e.g., "Based on the text you selected...").
5. Provide clear, educational explanations.
6. Be concise but thorough in your answers.
7. Cite the relevant information from your sources.

Important: Never make up information. If you're not sure based on the provided content, admit it.
""",
    model=model,
    tools=[retrieve, use_selected_text]
)

# ---------------------------------------------------
# Retry Logic for API Calls
# ---------------------------------------------------
def route_to_specialist(query: str, selected_text: Optional[str] = None) -> Literal["physics", "programming", "mathematics", "general"]:
    """Route query to the appropriate specialist based on keywords"""

    query_lower = query.lower()

    # Check for selected text - if present, use general agent with prioritized text
    if selected_text:
        return "general"

    # Programming keywords
    programming_keywords = [
        'code', 'program', 'algorithm', 'function', 'class', 'implement',
        'python', 'c++', 'java', 'ros', 'api', 'software', 'debug',
        'syntax', 'variable', 'loop', 'conditional', 'array', 'list'
    ]

    # Mathematics keywords
    mathematics_keywords = [
        'equation', 'formula', 'calculate', 'derivative', 'integral',
        'matrix', 'vector', 'trigonometry', 'geometry', 'algebra',
        'statistics', 'probability', 'optimization', 'linear algebra',
        'calculus', 'proof', 'theorem', 'mathematics'
    ]

    # Physics keywords
    physics_keywords = [
        'force', 'torque', 'momentum', 'energy', 'velocity', 'acceleration',
        'mass', 'gravity', 'friction', 'pressure', 'stress', 'strain',
        'mechanics', 'dynamics', 'kinematics', 'physics', 'motion',
        'balance', 'stability', 'vibration', 'wave', 'heat'
    ]

    # Count keyword matches
    programming_score = sum(1 for keyword in programming_keywords if keyword in query_lower)
    mathematics_score = sum(1 for keyword in mathematics_keywords if keyword in query_lower)
    physics_score = sum(1 for keyword in physics_keywords if keyword in query_lower)

    # Determine the best specialist
    scores = {
        "programming": programming_score,
        "mathematics": mathematics_score,
        "physics": physics_score
    }

    # Return the specialist with the highest score, or general if tie/zero
    max_score = max(scores.values())
    if max_score == 0:
        return "general"

    # Handle ties by returning the first with max score
    for specialist, score in scores.items():
        if score == max_score:
            return specialist

    return "general"

# ---------------------------------------------------
# Retry Logic for API Calls
# ---------------------------------------------------
async def run_with_retry(agent: Agent, user_input: str, selected_text: Optional[str] = None, max_retries: int = 3) -> Any:
    """
    Execute the agent with retry logic for handling rate limits
    """
    base_delay = 1.0
    max_delay = 10.0

    # Modify the input if we have selected text
    if selected_text:
        # Add context about selected text to guide the agent
        modified_input = f"""
User Question: {user_input}

User has selected the following text which is highly relevant to their question:
"{selected_text}"

Please address their question using this selected text as the primary source of information.
"""
    else:
        modified_input = user_input

    for attempt in range(max_retries):
        try:
            result = await Runner.run(agent, input=modified_input)
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
        message="AI Tutor API is running with User Text Selection support",
        details={
            "model": model.model,
            "embedding_model": embed_model,
            "collection": collection_name,
            "features": ["RAG", "User Text Selection", "Chat"]
        }
    )

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Comprehensive health check"""
    health_status = {
        "openai": False,
        "qdrant": False,
        "embedding_model": embed_model,
        "chat_model": model.model,
        "features": {
            "rag": True,
            "text_selection": True,
            "chat": True
        }
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
    Now supports user-selected text for prioritized responses
    """
    start_time = time.time()

    try:
        logger.info(f"Received message: {request.message[:100]}...")
        if request.selected_text:
            logger.info(f"User provided selected text: {request.selected_text[:100]}...")

        # Run the agent with retry logic and selected text
        result = await run_with_retry(agent, request.message, request.selected_text)

        # Extract sources from the agent's tool calls
        sources = []
        retrieval_count = 0
        used_selected_text = False

        if hasattr(result, 'raw_responses') and result.raw_responses:
            for response in result.raw_responses:
                if hasattr(response, 'tool_calls'):
                    for tool_call in response.tool_calls:
                        if tool_call.function.name == "retrieve":
                            # Regular retrieval
                            retrieved_texts = tool_call.function.result
                            if isinstance(retrieved_texts, list):
                                retrieval_count = len(retrieved_texts)
                                sources = [
                                    Source(
                                        text=text[:500] + "..." if len(text) > 500 else text,
                                        is_user_selected=False
                                    )
                                    for text in retrieved_texts
                                ]
                        elif tool_call.function.name == "use_selected_text":
                            # User-selected text was used
                            used_selected_text = True
                            retrieved_texts = tool_call.function.result
                            if isinstance(retrieved_texts, list):
                                retrieval_count = len(retrieved_texts)
                                for i, text in enumerate(retrieved_texts):
                                    is_user_selected = (i == 0)  # First result is the user selection
                                    sources.append(
                                        Source(
                                            text=text[:500] + "..." if len(text) > 500 else text,
                                            is_user_selected=is_user_selected,
                                            url="User Selection" if is_user_selected else None
                                        )
                                    )

        # Calculate response time
        response_time = time.time() - start_time
        logger.info(f"Response generated in {response_time:.2f}s with {retrieval_count} sources")

        return ChatResponse(
            response=result.final_output,
            sources=sources,
            conversation_id=request.conversation_id,
            session_id=request.session_id,
            model_used=model.model,
            retrieval_count=retrieval_count,
            used_selected_text=used_selected_text
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

@app.post("/chat/specialist", response_model=SpecialistResponse)
async def chat_with_specialist(request: SpecialistRequest):
    """
    Chat endpoint with automatic specialist routing or manual selection
    """
    start_time = time.time()

    try:
        logger.info(f"Received specialist request: {request.message[:100]}...")

        # Determine which specialist to use
        if request.specialist_type == "general":
            # Auto-route based on content
            specialist_type = route_to_specialist(request.message, request.selected_text)
        else:
            specialist_type = request.specialist_type

        logger.info(f"Selected specialist: {specialist_type}")

        # Get the appropriate agent
        agent = specialist_agents[specialist_type]

        # Run the agent with retry logic
        result = await run_with_retry(agent, request.message, request.selected_text)

        # Calculate response time
        response_time = time.time() - start_time
        logger.info(f"Specialist response generated in {response_time:.2f}s")

        return SpecialistResponse(
            response=result.final_output,
            specialist_type=specialist_type,
            conversation_id=request.conversation_id,
            session_id=request.session_id,
            model_used=model.model,
            confidence=0.85  # Placeholder confidence score
        )

    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"Specialist chat endpoint error: {e}")
        raise HTTPException(status_code=500, detail="Failed to process specialist chat request")

@app.get("/specialists")
async def get_specialists():
    """Get list of available specialist agents"""
    specialists_info = {
        "specialists": [
            {
                "type": "physics",
                "name": "Physics Tutor",
                "description": "Expert in mechanics, forces, energy, and control theory",
                "keywords": ["force", "torque", "momentum", "energy", "mechanics"]
            },
            {
                "type": "programming",
                "name": "Programming Tutor",
                "description": "Expert in robotics programming, algorithms, and software architecture",
                "keywords": ["code", "algorithm", "python", "implement", "software"]
            },
            {
                "type": "mathematics",
                "name": "Mathematics Tutor",
                "description": "Expert in linear algebra, calculus, and optimization for robotics",
                "keywords": ["equation", "matrix", "calculate", "geometry", "algebra"]
            },
            {
                "type": "general",
                "name": "General AI Tutor",
                "description": "General assistant that can help with any topic and prioritizes selected text",
                "keywords": ["general", "help", "explain", "overview"]
            }
        ],
        "auto_routing": True,
        "supported_languages": ["python", "c++", "javascript"]
    }
    return specialists_info

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
    parser = argparse.ArgumentParser(description="AI Tutor API Server with User Text Selection")
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

    logger.info(f"Starting AI Tutor API server on port {args.port} with User Text Selection support...")
    uvicorn.run(**uvicorn_config)