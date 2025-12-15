"""
FastAPI RAG Chatbot Backend with Authentication (Final Fix)
===========================================================
Fixed: Pydantic Validation Error for JSONB fields
"""

from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, EmailStr, field_validator
from typing import List, Optional, Dict, Any, Literal
import asyncio
import logging
import os
import random
import time
import json
from contextlib import asynccontextmanager
from datetime import datetime, timedelta
from uuid import UUID

from dotenv import load_dotenv
from agents import Agent, Runner, OpenAIChatCompletionsModel
from agents import set_tracing_disabled, function_tool
from openai import AsyncOpenAI, APIError

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
import asyncpg

# Auth dependencies
import bcrypt
from jose import JWTError, jwt

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
load_dotenv(Path(__file__).parent / '.env')
set_tracing_disabled(disabled=True)

# JWT Configuration
SECRET_KEY = os.getenv("JWT_SECRET_KEY", "your-secret-key-change-in-production")
if not SECRET_KEY:
    raise ValueError("JWT_SECRET_KEY is missing in environment variables!")

ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_DAYS = 7

security = HTTPBearer(auto_error=False)

# ---------------------------------------------------
# Auth Helper Functions
# ---------------------------------------------------
def hash_password(password: str) -> str:
    """Hash password using bcrypt"""
    return bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')

def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify password against hash"""
    return bcrypt.checkpw(plain_password.encode('utf-8'), hashed_password.encode('utf-8'))

def create_access_token(data: dict) -> str:
    """Create JWT access token"""
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(days=ACCESS_TOKEN_EXPIRE_DAYS)
    
    # Ensure any UUID objects in data are converted to strings for JWT
    for k, v in to_encode.items():
        if isinstance(v, UUID):
            to_encode[k] = str(v)
            
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

def decode_access_token(token: str) -> Optional[Dict]:
    """Decode JWT token"""
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return payload
    except JWTError:
        return None

# ---------------------------------------------------
# Database Helper Class (Robust)
# ---------------------------------------------------
class NeonDB:
    """Helper class for Neon database operations - UUID & JSON Robust"""

    def __init__(self):
        self.connection_string = os.getenv('NEON_DATABASE_URL')
        self.pool = None

    async def initialize(self):
        """Initialize connection pool"""
        if not self.connection_string:
            logger.warning("NEON_DATABASE_URL not set")
            return False

        try:
            self.pool = await asyncpg.create_pool(
                self.connection_string,
                min_size=2,
                max_size=10,
                command_timeout=60
            )
            await self.create_tables()
            return True
        except Exception as e:
            logger.error(f"Failed to connect to Neon: {e}")
            return False

    def _process_row(self, row) -> Optional[Dict]:
        """Helper to safely convert UUIDs to strings and JSON strings to dicts"""
        if not row:
            return None
        
        data = dict(row)
        
        # 1. Convert UUIDs to strings
        for col in ['id', 'user_id']:
            if col in data and isinstance(data[col], (UUID, object)):
                data[col] = str(data[col])

        # 2. Ensure JSON fields are Dicts (parse if they are strings)
        # This is a backup; Pydantic validator will also handle this.
        json_fields = ['software_background', 'hardware_background', 'preferences', 'metadata', 'event_data']
        for field in json_fields:
            if field in data:
                val = data[field]
                if isinstance(val, str):
                    try:
                        data[field] = json.loads(val)
                    except json.JSONDecodeError:
                        data[field] = {}
                elif val is None:
                    data[field] = {}
                    
        return data

    async def create_tables(self):
        """Create necessary tables matching the UUID schema"""
        if not self.pool:
            return

        try:
            async with self.pool.acquire() as conn:
                # Users table (UUID Primary Key)
                await conn.execute("""
                    CREATE TABLE IF NOT EXISTS users (
                        id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                        email VARCHAR(255) UNIQUE NOT NULL,
                        password_hash VARCHAR(255) NOT NULL,
                        name TEXT,
                        software_background JSONB DEFAULT '{}'::jsonb,
                        hardware_background JSONB DEFAULT '{}'::jsonb,
                        preferences JSONB DEFAULT '{}'::jsonb,
                        profile_completed BOOLEAN DEFAULT FALSE,
                        last_active TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
                        created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                
                # User sessions table
                await conn.execute("""
                    CREATE TABLE IF NOT EXISTS user_sessions (
                        session_id VARCHAR(255) PRIMARY KEY,
                        user_id UUID REFERENCES users(id) ON DELETE CASCADE,
                        message_count INTEGER DEFAULT 0,
                        last_active TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                
                # Conversations table
                await conn.execute("""
                    CREATE TABLE IF NOT EXISTS conversations (
                        id SERIAL PRIMARY KEY,
                        conversation_id VARCHAR(255) NOT NULL,
                        session_id VARCHAR(255) REFERENCES user_sessions(session_id),
                        user_id UUID REFERENCES users(id) ON DELETE CASCADE,
                        user_message TEXT NOT NULL,
                        assistant_response TEXT NOT NULL,
                        specialist_type VARCHAR(50),
                        confidence FLOAT,
                        metadata JSONB,
                        timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                
                # Analytics table
                await conn.execute("""
                    CREATE TABLE IF NOT EXISTS analytics (
                        id SERIAL PRIMARY KEY,
                        event_type VARCHAR(100) NOT NULL,
                        user_id UUID REFERENCES users(id) ON DELETE SET NULL,
                        session_id VARCHAR(255),
                        event_data JSONB,
                        timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                logger.info("Database tables verified successfully")
        except Exception as e:
            logger.error(f"Error creating tables: {e}")

    # User Management
    async def create_user(
        self,
        email: str,
        password: str,
        name: str,
        software_background: Optional[Dict] = None,
        hardware_background: Optional[Dict] = None
    ) -> Optional[Dict]:
        if not self.pool:
            return None

        try:
            password_hash = hash_password(password)
            sw_bg_json = json.dumps(software_background) if software_background else '{}'
            hw_bg_json = json.dumps(hardware_background) if hardware_background else '{}'

            async with self.pool.acquire() as conn:
                row = await conn.fetchrow(
                    """
                    INSERT INTO users (email, password_hash, name, software_background, hardware_background)
                    VALUES ($1, $2, $3, $4::jsonb, $5::jsonb)
                    RETURNING id, email, name, software_background, hardware_background, created_at
                    """,
                    email, password_hash, name, sw_bg_json, hw_bg_json
                )
                return self._process_row(row)
        except Exception as e:
            logger.error(f"Error creating user: {e}")
            return None

    async def get_user_by_email(self, email: str) -> Optional[Dict]:
        if not self.pool:
            return None

        try:
            async with self.pool.acquire() as conn:
                row = await conn.fetchrow("SELECT * FROM users WHERE email = $1", email)
                return self._process_row(row)
        except Exception as e:
            logger.error(f"Error getting user: {e}")
            return None

    async def get_user_by_id(self, user_id: str) -> Optional[Dict]:
        if not self.pool:
            return None

        try:
            async with self.pool.acquire() as conn:
                row = await conn.fetchrow(
                    "SELECT id, email, name, software_background, hardware_background, preferences, created_at FROM users WHERE id = $1",
                    user_id
                )
                return self._process_row(row)
        except Exception as e:
            logger.error(f"Error getting user by ID: {e}")
            return None

    # Conversation Management
    async def save_conversation(self, conversation_id, session_id, user_id, user_message, assistant_response, specialist_type=None, confidence=None, metadata=None):
        if not self.pool:
            return False
        try:
            async with self.pool.acquire() as conn:
                await conn.execute(
                    """
                    INSERT INTO user_sessions (session_id, user_id, message_count)
                    VALUES ($1, $2, 1)
                    ON CONFLICT (session_id) DO UPDATE SET
                        message_count = user_sessions.message_count + 1,
                        last_active = CURRENT_TIMESTAMP
                    """,
                    session_id, user_id
                )
                await conn.execute(
                    """
                    INSERT INTO conversations
                    (conversation_id, session_id, user_id, user_message, assistant_response,
                     specialist_type, confidence, metadata)
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
                    """,
                    conversation_id, session_id, user_id, user_message, 
                    assistant_response, specialist_type, confidence,
                    json.dumps(metadata) if metadata else None
                )
                return True
        except Exception as e:
            logger.error(f"Error saving conversation: {e}")
            return False

    async def get_conversation_history(self, conversation_id: str, limit: int = 10) -> List[Dict]:
        if not self.pool:
            return []
        try:
            async with self.pool.acquire() as conn:
                rows = await conn.fetch(
                    """
                    SELECT user_message, assistant_response, specialist_type,
                           confidence, timestamp
                    FROM conversations
                    WHERE conversation_id = $1
                    ORDER BY timestamp ASC
                    LIMIT $2
                    """,
                    conversation_id, limit
                )
                return [dict(row) for row in rows]
        except Exception as e:
            logger.error(f"Error retrieving conversation: {e}")
            return []

    async def get_user_conversations(self, user_id: str, limit: int = 20) -> List[Dict]:
        if not self.pool:
            return []
        try:
            async with self.pool.acquire() as conn:
                rows = await conn.fetch(
                    """
                    SELECT DISTINCT ON (conversation_id) 
                        conversation_id, user_message, timestamp
                    FROM conversations
                    WHERE user_id = $1
                    ORDER BY conversation_id, timestamp DESC
                    LIMIT $2
                    """,
                    user_id, limit
                )
                return [dict(row) for row in rows]
        except Exception as e:
            logger.error(f"Error retrieving user conversations: {e}")
            return []

    async def log_event(self, event_type: str, user_id: Optional[str] = None, session_id: Optional[str] = None, data: Optional[Dict[str, Any]] = None):
        if not self.pool:
            return
        try:
            async with self.pool.acquire() as conn:
                await conn.execute(
                    """
                    INSERT INTO analytics (event_type, user_id, session_id, event_data)
                    VALUES ($1, $2, $3, $4::jsonb)
                    """,
                    event_type, user_id, session_id, json.dumps(data) if data else None
                )
        except Exception as e:
            logger.error(f"Error logging event: {e}")

    async def close(self):
        if self.pool:
            await self.pool.close()

# Global database instance
db = NeonDB()

# ---------------------------------------------------
# Auth Dependency
# ---------------------------------------------------
async def get_current_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security)
) -> Optional[Dict]:
    """Get current authenticated user (optional)"""
    if not credentials:
        return None
    
    token = credentials.credentials
    payload = decode_access_token(token)
    
    if not payload:
        return None
    
    user_id = payload.get("user_id")
    if not user_id:
        return None
    
    user = await db.get_user_by_id(user_id)
    return user

async def require_auth(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security)
) -> Dict:
    """Require authentication (returns user or raises 401)"""
    if not credentials:
        raise HTTPException(status_code=401, detail="Authentication required")
    
    token = credentials.credentials
    payload = decode_access_token(token)
    
    if not payload:
        raise HTTPException(status_code=401, detail="Invalid or expired token")
    
    user_id = payload.get("user_id")
    if not user_id:
        raise HTTPException(status_code=401, detail="Invalid token payload")
    
    user = await db.get_user_by_id(user_id)
    if not user:
        raise HTTPException(status_code=401, detail="User not found")
    
    return user

# ---------------------------------------------------
# Database Lifecycle Management
# ---------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    await db.initialize()
    logger.info("Neon database connected successfully")
    yield
    # Shutdown
    await db.close()
    logger.info("Database connection closed")

# ---------------------------------------------------
# FastAPI App Configuration
# ---------------------------------------------------
app = FastAPI(
    title="AI Tutor API with Authentication",
    description="RAG-based AI tutor with Better-Auth authentication and chat memory",
    version="3.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------
# Model and Client Initialization
# ---------------------------------------------------
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    raise ValueError("Missing OPENAI_API_KEY in environment variables")

chat_client = AsyncOpenAI(api_key=openai_api_key)
embedding_client = AsyncOpenAI(api_key=openai_api_key)

model = OpenAIChatCompletionsModel(
    model="gpt-4o-mini",
    openai_client=chat_client
)

qdrant_url = os.getenv("QDRANT_URL")
qdrant_api_key = os.getenv("QDRANT_API_KEY")
collection_name = os.getenv("COLLECTION_NAME", "humanoid_ai_book_openai")
embed_model = os.getenv("EMBED_MODEL", "text-embedding-3-small")

if not qdrant_url or not qdrant_api_key:
    raise ValueError("Missing QDRANT_URL or QDRANT_API_KEY")

qdrant = QdrantClient(url=qdrant_url, api_key=qdrant_api_key)

# ---------------------------------------------------
# Pydantic Schemas (FIXED WITH VALIDATOR)
# ---------------------------------------------------
class SignUpRequest(BaseModel):
    email: EmailStr
    password: str
    name: str
    software_background: Optional[Dict[str, Any]] = {}
    hardware_background: Optional[Dict[str, Any]] = {}

class SignInRequest(BaseModel):
    email: EmailStr
    password: str

class UserProfile(BaseModel):
    id: str
    email: str
    name: Optional[str] = None
    software_background: Optional[Dict[str, Any]] = {}
    hardware_background: Optional[Dict[str, Any]] = {}
    created_at: datetime

    # --- THIS IS THE CRITICAL FIX ---
    @field_validator('software_background', 'hardware_background', mode='before')
    @classmethod
    def parse_json_fields(cls, v):
        if isinstance(v, str):
            try:
                # If database returns a string, convert it to a dict
                return json.loads(v)
            except ValueError:
                return {}
        return v
    # --------------------------------

class AuthResponse(BaseModel):
    token: str
    user: UserProfile

class ChatRequest(BaseModel):
    message: str
    conversation_id: Optional[str] = None
    session_id: Optional[str] = None
    stream: bool = False
    selected_text: Optional[str] = None

class Source(BaseModel):
    text: str
    url: Optional[str] = None
    score: Optional[float] = None
    is_user_selected: Optional[bool] = False

class ChatResponse(BaseModel):
    response: str
    sources: List[Source] = []
    conversation_id: str
    session_id: str
    model_used: str
    retrieval_count: int
    used_selected_text: bool = False

class HealthResponse(BaseModel):
    status: str
    message: str
    details: Optional[Dict[str, Any]] = None

class SpecialistRequest(BaseModel):
    message: str
    specialist_type: Literal["physics", "programming", "mathematics", "general"] = "general"
    conversation_id: Optional[str] = None
    session_id: Optional[str] = None
    selected_text: Optional[str] = None

class SpecialistResponse(BaseModel):
    response: str
    specialist_type: str
    conversation_id: str
    session_id: str
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
    """Retrieve relevant textbook excerpts from Qdrant database"""
    try:
        embedding = await get_embedding(query)
        search_result = qdrant.query_points(
            collection_name=collection_name,
            query=embedding,
            limit=5,
            with_payload=True,
            with_vectors=False
        )

        retrieved_texts = []
        for point in search_result.points:
            if hasattr(point, 'payload') and point.payload:
                text = point.payload.get('text', '')
                if text:
                    retrieved_texts.append(text)

        logger.info(f"Retrieved {len(retrieved_texts)} documents")
        return retrieved_texts

    except Exception as e:
        logger.error(f"Retrieval error: {e}")
        return []

@function_tool
async def use_selected_text(text: str, query: str) -> List[str]:
    """Process user-selected text with priority"""
    try:
        results = [text]
        embedding = await get_embedding(text)

        search_result = qdrant.query_points(
            collection_name=collection_name,
            query=embedding,
            limit=3,
            with_payload=True,
            with_vectors=False,
            score_threshold=0.8
        )

        for point in search_result.points:
            if hasattr(point, 'payload') and point.payload:
                db_text = point.payload.get('text', '')
                if db_text and db_text != text:
                    results.append(db_text)

        logger.info(f"Processed selected text with {len(results)-1} supplementary results")
        return results

    except Exception as e:
        logger.error(f"Error processing selected text: {e}")
        return [text]

@function_tool
async def generate_code_example(concept: str, language: str = "python") -> str:
    """Generate a practical code example"""
    return f"# Example code for {concept} in {language}\n# (Generated by AI Tutor)"

@function_tool
async def create_practice_question(topic: str, difficulty: str = "medium") -> str:
    """Create a practice question"""
    return f"Q: Practice question about {topic} ({difficulty}) - Coming soon!"

# ---------------------------------------------------
# AI Agent Configuration
# ---------------------------------------------------
def create_specialist_agents() -> Dict[str, Agent]:
    """Create specialized agents"""
    
    physics_agent = Agent(
        name="Physics Tutor",
        instructions="""You are an expert in physics for robotics. Use retrieve tool for textbook content.""",
        model=model,
        tools=[retrieve, use_selected_text, generate_code_example, create_practice_question]
    )

    programming_agent = Agent(
        name="Programming Tutor",
        instructions="""You are an expert in robotics programming. Use retrieve tool for examples.""",
        model=model,
        tools=[retrieve, use_selected_text, generate_code_example, create_practice_question]
    )

    math_agent = Agent(
        name="Mathematics Tutor",
        instructions="""You are an expert in mathematics for robotics. Provide step-by-step derivations.""",
        model=model,
        tools=[retrieve, use_selected_text, generate_code_example, create_practice_question]
    )

    general_agent = Agent(
        name="General AI Tutor",
        instructions="""You are a general AI tutor. Prioritize user-selected text when provided.""",
        model=model,
        tools=[retrieve, use_selected_text]
    )

    return {
        "physics": physics_agent,
        "programming": programming_agent,
        "mathematics": math_agent,
        "general": general_agent
    }

specialist_agents = create_specialist_agents()

agent = Agent(
    name="AI Tutor",
    instructions="""You are an expert AI tutor. Use selected text when provided, otherwise retrieve relevant information.""",
    model=model,
    tools=[retrieve, use_selected_text]
)

# ---------------------------------------------------
# Helper Functions
# ---------------------------------------------------
def route_to_specialist(query: str, selected_text: Optional[str] = None) -> Literal["physics", "programming", "mathematics", "general"]:
    """Route query to appropriate specialist"""
    if selected_text:
        return "general"
    
    query_lower = query.lower()
    
    programming_kw = ['code', 'program', 'algorithm', 'function', 'implement']
    mathematics_kw = ['equation', 'formula', 'calculate', 'matrix', 'vector']
    physics_kw = ['force', 'torque', 'momentum', 'energy', 'dynamics']
    
    p_score = sum(1 for kw in programming_kw if kw in query_lower)
    m_score = sum(1 for kw in mathematics_kw if kw in query_lower)
    ph_score = sum(1 for kw in physics_kw if kw in query_lower)
    
    scores = {"programming": p_score, "mathematics": m_score, "physics": ph_score}
    max_score = max(scores.values())
    
    if max_score == 0:
        return "general"
    
    for specialist, score in scores.items():
        if score == max_score:
            return specialist
    
    return "general"

async def run_with_retry(agent: Agent, user_input: str, selected_text: Optional[str] = None, max_retries: int = 3):
    """Execute agent with retry logic"""
    base_delay = 1.0
    max_delay = 10.0

    if selected_text:
        modified_input = f"""
User Question: {user_input}

User has selected the following text:
"{selected_text}"

Please address their question using this selected text as the primary source.
"""
    else:
        modified_input = user_input

    for attempt in range(max_retries):
        try:
            result = await Runner.run(agent, input=modified_input)
            return result

        except APIError as e:
            if "rate_limit" in str(e).lower() or "429" in str(e):
                if attempt < max_retries - 1:
                    delay = min(base_delay * (2 ** attempt), max_delay)
                    await asyncio.sleep(delay)
                    continue
                else:
                    raise HTTPException(status_code=429, detail="Rate limit exceeded")
            else:
                logger.error(f"OpenAI API error: {e}")
                raise HTTPException(status_code=500, detail="AI service error")

        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            if attempt < max_retries - 1:
                await asyncio.sleep(base_delay)
                continue
            raise HTTPException(status_code=500, detail="Internal server error")

    raise HTTPException(status_code=500, detail="Failed after retries")

def format_conversation_context(history: List[Dict]) -> str:
    """Format conversation history as context"""
    if not history:
        return ""
    
    context = "Previous conversation:\n\n"
    for msg in history:
        context += f"User: {msg['user_message']}\n"
        context += f"Assistant: {msg['assistant_response']}\n\n"
    
    return context

# ---------------------------------------------------
# Authentication Endpoints
# ---------------------------------------------------
@app.post("/api/auth/signup", response_model=AuthResponse)
async def signup(request: SignUpRequest):
    """Create new user account"""
    try:
        existing_user = await db.get_user_by_email(request.email)
        if existing_user:
            raise HTTPException(status_code=400, detail="Email already registered")
        
        user = await db.create_user(
            email=request.email,
            password=request.password,
            name=request.name,
            software_background=request.software_background,
            hardware_background=request.hardware_background
        )
        
        if not user:
            raise HTTPException(status_code=500, detail="Failed to create user")
        
        token = create_access_token({"user_id": user["id"]})
        await db.log_event("user_signup", user_id=user["id"])
        
        # Pydantic will now automatically handle string parsing via the field_validator
        return AuthResponse(
            token=token,
            user=UserProfile(
                id=user["id"],
                email=user["email"],
                name=user["name"],
                software_background=user.get("software_background"),
                hardware_background=user.get("hardware_background"),
                created_at=user["created_at"]
            )
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Signup error: {e}")
        raise HTTPException(status_code=500, detail="Signup failed")

@app.post("/api/auth/signin", response_model=AuthResponse)
async def signin(request: SignInRequest):
    """Sign in user"""
    try:
        user = await db.get_user_by_email(request.email)
        if not user:
            raise HTTPException(status_code=401, detail="Invalid email or password")
        
        if not verify_password(request.password, user["password_hash"]):
            raise HTTPException(status_code=401, detail="Invalid email or password")
        
        token = create_access_token({"user_id": user["id"]})
        await db.log_event("user_signin", user_id=user["id"])
        
        return AuthResponse(
            token=token,
            user=UserProfile(
                id=user["id"],
                email=user["email"],
                name=user["name"],
                software_background=user.get("software_background"),
                hardware_background=user.get("hardware_background"),
                created_at=user["created_at"]
            )
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Signin error: {e}")
        raise HTTPException(status_code=500, detail="Signin failed")

@app.post("/api/auth/signout")
async def signout(user: Dict = Depends(require_auth)):
    """Sign out user"""
    await db.log_event("user_signout", user_id=user["id"])
    return {"message": "Signed out successfully"}

@app.get("/api/auth/me", response_model=UserProfile)
async def get_me(user: Dict = Depends(require_auth)):
    """Get current user profile"""
    # The 'user' dict here comes from get_user_by_id
    # Pydantic will validate this dict against UserProfile schema
    return UserProfile(
        id=user["id"],
        email=user["email"],
        name=user["name"],
        software_background=user.get("software_background"),
        hardware_background=user.get("hardware_background"),
        created_at=user["created_at"]
    )

# ---------------------------------------------------
# Chat Endpoints (with Auth Support)
# ---------------------------------------------------
@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest, current_user: Optional[Dict] = Depends(get_current_user)):
    """Main chat endpoint with optional authentication"""
    start_time = time.time()

    try:
        conversation_id = request.conversation_id or f"conv_{datetime.now().timestamp()}"
        session_id = request.session_id or f"session_{conversation_id}"
        user_id = current_user["id"] if current_user else None
        
        # Get conversation history
        history = await db.get_conversation_history(conversation_id, limit=5)
        context = format_conversation_context(history)
        
        # Add user background
        if current_user:
            bg_info = []
            if current_user.get("software_background"):
                # Handle both dict and potentially parsed string cases safely
                sw = current_user['software_background']
                if isinstance(sw, str):
                    try: sw = json.loads(sw)
                    except: sw = {}
                bg_info.append(f"Software: {sw}")

            if current_user.get("hardware_background"):
                hw = current_user['hardware_background']
                if isinstance(hw, str):
                    try: hw = json.loads(hw)
                    except: hw = {}
                bg_info.append(f"Hardware: {hw}")
                
            if bg_info:
                context += "User background: " + ", ".join(bg_info) + "\n\n"
        
        full_input = context + f"Current question: {request.message}"
        result = await run_with_retry(agent, full_input, request.selected_text)

        # Extract sources
        sources = []
        retrieval_count = 0
        used_selected_text = False

        if hasattr(result, 'raw_responses') and result.raw_responses:
            for response in result.raw_responses:
                if hasattr(response, 'tool_calls'):
                    for tool_call in response.tool_calls:
                        if tool_call.function.name == "retrieve":
                            retrieved_texts = tool_call.function.result
                            if isinstance(retrieved_texts, list):
                                retrieval_count = len(retrieved_texts)
                                sources = [Source(text=t[:500]+"..." if len(t)>500 else t, is_user_selected=False) for t in retrieved_texts]
                        elif tool_call.function.name == "use_selected_text":
                            used_selected_text = True
                            retrieved_texts = tool_call.function.result
                            if isinstance(retrieved_texts, list):
                                retrieval_count = len(retrieved_texts)
                                for i, text in enumerate(retrieved_texts):
                                    is_user_selected = (i == 0)
                                    sources.append(Source(
                                        text=text[:500]+"..." if len(text)>500 else text,
                                        is_user_selected=is_user_selected,
                                        url="User Selection" if is_user_selected else None
                                    ))

        # Save conversation
        await db.save_conversation(
            conversation_id=conversation_id,
            session_id=session_id,
            user_id=user_id,
            user_message=request.message,
            assistant_response=result.final_output,
            metadata={"selected_text": request.selected_text, "response_time": time.time()-start_time, "retrieval_count": retrieval_count}
        )

        logger.info(f"Response in {time.time()-start_time:.2f}s with {retrieval_count} sources")

        return ChatResponse(
            response=result.final_output,
            sources=sources,
            conversation_id=conversation_id,
            session_id=session_id,
            model_used=model.model,
            retrieval_count=retrieval_count,
            used_selected_text=used_selected_text
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Chat error: {e}")
        raise HTTPException(status_code=500, detail="Failed to process chat")

@app.post("/chat/specialist", response_model=SpecialistResponse)
async def chat_with_specialist(request: SpecialistRequest, current_user: Optional[Dict] = Depends(get_current_user)):
    """Chat with specialist agent"""
    start_time = time.time()

    try:
        conversation_id = request.conversation_id or f"conv_{datetime.now().timestamp()}"
        session_id = request.session_id or f"session_{conversation_id}"
        user_id = current_user["id"] if current_user else None

        # Determine specialist
        if request.specialist_type == "general":
            specialist_type = route_to_specialist(request.message, request.selected_text)
        else:
            specialist_type = request.specialist_type

        # Get history and context
        history = await db.get_conversation_history(conversation_id, limit=5)
        context = format_conversation_context(history)
        
        # Add user background
        if current_user:
            bg_info = []
            if current_user.get("software_background"):
                bg_info.append(f"Software: {current_user['software_background']}")
            if current_user.get("hardware_background"):
                bg_info.append(f"Hardware: {current_user['hardware_background']}")
            if bg_info:
                context += "User background: " + ", ".join(bg_info) + "\n\n"

        full_input = context + f"Current question: {request.message}"

        # Run agent
        agent_to_use = specialist_agents[specialist_type]
        result = await run_with_retry(agent_to_use, full_input, request.selected_text)

        # Save conversation
        await db.save_conversation(
            conversation_id=conversation_id,
            session_id=session_id,
            user_id=user_id,
            user_message=request.message,
            assistant_response=result.final_output,
            specialist_type=specialist_type,
            confidence=0.85,
            metadata={"selected_text": request.selected_text, "response_time": time.time()-start_time}
        )

        return SpecialistResponse(
            response=result.final_output,
            specialist_type=specialist_type,
            conversation_id=conversation_id,
            session_id=session_id,
            model_used=model.model,
            confidence=0.85
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Specialist chat error: {e}")
        raise HTTPException(status_code=500, detail="Failed to process specialist chat")

@app.post("/chat/continue")
async def continue_chat(request: Dict[str, Any], current_user: Optional[Dict] = Depends(get_current_user)):
    """Continue an existing conversation"""
    try:
        message = request.get("message")
        if not message:
            raise HTTPException(status_code=400, detail="Message is required")

        conversation_id = request.get("conversation_id") or f"conv_{datetime.now().timestamp()}"
        specialist_type = request.get("specialist_type", "general")
        selected_text = request.get("selected_text")
        user_id = current_user["id"] if current_user else None

        # Auto-route if general
        if specialist_type == "general":
            specialist_type = route_to_specialist(message, selected_text)

        # Get conversation history
        history = await db.get_conversation_history(conversation_id, limit=5)
        context = format_conversation_context(history)
        
        # Add user background
        if current_user:
            bg_info = []
            if current_user.get("software_background"):
                bg_info.append(f"Software: {current_user['software_background']}")
            if current_user.get("hardware_background"):
                bg_info.append(f"Hardware: {current_user['hardware_background']}")
            if bg_info:
                context += "User background: " + ", ".join(bg_info) + "\n\n"

        full_input = context + f"Current question: {message}"

        # Get agent and run
        agent_to_use = specialist_agents[specialist_type]
        result = await run_with_retry(agent_to_use, full_input, selected_text)
        session_id = f"session_{conversation_id}"

        # Save conversation
        await db.save_conversation(
            conversation_id=conversation_id,
            session_id=session_id,
            user_id=user_id,
            user_message=message,
            assistant_response=result.final_output,
            specialist_type=specialist_type,
            confidence=0.85,
            metadata={"is_continuation": len(history)>0, "history_count": len(history), "selected_text": selected_text}
        )

        return {
            "response": result.final_output,
            "specialist_type": specialist_type,
            "conversation_id": conversation_id,
            "session_id": session_id,
            "model_used": model.model,
            "confidence": 0.85,
            "is_continuation": len(history) > 0,
            "history_count": len(history)
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Continue chat error: {e}")
        raise HTTPException(status_code=500, detail="Failed to continue chat")

# ---------------------------------------------------
# Utility Endpoints
# ---------------------------------------------------
@app.get("/", response_model=HealthResponse)
async def root():
    """Root endpoint"""
    return HealthResponse(
        status="healthy",
        message="AI Tutor API with Authentication is running",
        details={
            "model": model.model,
            "embedding_model": embed_model,
            "collection": collection_name,
            "features": ["RAG", "Authentication", "User Text Selection", "Chat Memory", "UUID Support"]
        }
    )

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check"""
    health_status = {
        "openai": False,
        "qdrant": False,
        "database": False,
        "embedding_model": embed_model,
        "chat_model": model.model
    }

    try:
        test_embedding = await get_embedding("test")
        if test_embedding:
            health_status["openai"] = True
    except Exception as e:
        logger.error(f"OpenAI health check failed: {e}")

    try:
        collections = qdrant.get_collections()
        if any(c.name == collection_name for c in collections.collections):
            health_status["qdrant"] = True
    except Exception as e:
        logger.error(f"Qdrant health check failed: {e}")

    try:
        if db.pool:
            health_status["database"] = True
    except Exception as e:
        logger.error(f"Database health check failed: {e}")

    all_healthy = all([health_status["openai"], health_status["qdrant"], health_status["database"]])

    return HealthResponse(
        status="healthy" if all_healthy else "degraded",
        message="All services operational" if all_healthy else "Some services unavailable",
        details=health_status
    )

@app.get("/conversation/{conversation_id}")
async def get_conversation_history_endpoint(conversation_id: str, limit: int = 10, current_user: Optional[Dict] = Depends(get_current_user)):
    """Get conversation history"""
    try:
        history = await db.get_conversation_history(conversation_id, limit)
        return {"conversation_id": conversation_id, "history": history}
    except Exception as e:
        logger.error(f"Error retrieving conversation: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve conversation")

@app.get("/my-conversations")
async def get_my_conversations(limit: int = 20, current_user: Dict = Depends(require_auth)):
    """Get all conversations for authenticated user"""
    try:
        conversations = await db.get_user_conversations(current_user["id"], limit)
        return {"conversations": conversations}
    except Exception as e:
        logger.error(f"Error retrieving user conversations: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve conversations")

@app.get("/specialists")
async def get_specialists():
    """Get list of available specialists"""
    return {
        "specialists": [
            {"type": "physics", "name": "Physics Tutor", "description": "Expert in mechanics, forces, energy, control theory"},
            {"type": "programming", "name": "Programming Tutor", "description": "Expert in robotics programming and algorithms"},
            {"type": "mathematics", "name": "Mathematics Tutor", "description": "Expert in linear algebra, calculus, optimization"},
            {"type": "general", "name": "General AI Tutor", "description": "General assistant for all topics"}
        ]
    }

@app.get("/collection/stats")
async def get_collection_stats():
    """Get Qdrant collection statistics"""
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
        raise HTTPException(status_code=404, detail="Collection not found")

@app.delete("/collection")
async def clear_collection(current_user: Dict = Depends(require_auth)):
    """Clear collection (authenticated only)"""
    try:
        qdrant.delete_collection(collection_name)
        vector_size = 1536 if embed_model == "text-embedding-3-small" else 3072
        qdrant.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE)
        )
        return {"message": f"Collection '{collection_name}' cleared and recreated"}
    except Exception as e:
        logger.error(f"Error clearing collection: {e}")
        raise HTTPException(status_code=500, detail="Failed to clear collection")

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    logger.info(f"Starting AI Tutor API server with authentication on port {port}...")
    uvicorn.run(app, host="0.0.0.0", port=port)