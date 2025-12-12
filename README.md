# RAG Chatbot - Physical AI & Humanoid Robotics

An intelligent chatbot system with RAG (Retrieval-Augmented Generation) capabilities for the Physical AI & Humanoid Robotics textbook.

## =€ Features

- **Book-Aware Specialist Agents**: Physics, Programming, Mathematics, and General AI tutors with deep knowledge of the textbook
- **Auto-Routing**: Automatically selects the appropriate specialist based on query keywords
- **Real-Time Responses**: Fast response times with confidence scoring
- **Text Selection Support**: Prioritizes user-selected text for contextual answers
- **Beautiful UI**: Modern, responsive chat widget with code highlighting
- **Database Integration**: 675+ documents from the textbook indexed in Qdrant Cloud

## =Ë Essential Files for Deployment

### Backend
- `main.py` - Main FastAPI application with all endpoints
- `requirements.txt` - Python dependencies
- `.env` - Environment variables (API keys, database credentials)
- `pyproject.toml` - Project configuration

### Frontend
- `src/theme/` - React components
- `styles.css` - Styling for the chat widget

### Configuration
- `.python-version` - Python version specification
- `uv.lock` - Dependency lock file

## =à Installation & Setup

### Prerequisites
- Python 3.11+
- Node.js 16+
- OpenAI API key
- Qdrant Cloud account

### Backend Setup

1. **Clone and navigate to the project**:
   ```bash
   cd rag-chatbot-render
   ```

2. **Create virtual environment**:
   ```bash
   python -m venv .venv
   .venv\Scripts\activate  # Windows
   source .venv/bin/activate  # Linux/Mac
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**:
   Create `.env` file with:
   ```env
   OPENAI_API_KEY=your_openai_api_key_here
   QDRANT_URL=your_qdrant_cloud_url
   QDRANT_API_KEY=your_qdrant_api_key
   COLLECTION_NAME=humanoid_ai_book_openai
   EMBED_MODEL=text-embedding-3-small
   ```

5. **Run the server**:
   ```bash
   python main.py
   ```

   The server will start on `http://localhost:8000`

### Frontend Setup

1. **Navigate to your React/Docusaurus project**
2. **Copy the ChatWidget component**:
   - Copy `src/theme/` to your project
   - Copy `styles.css` to your styles

3. **Import the component** in your layout or page:
   ```javascript
   import ChatWidget from './theme/ChatWidget';
   ```

4. **Add to your layout**:
   ```jsx
   <ChatWidget />
   ```

## =Ú API Endpoints

### Health Check
- `GET /` - Basic health info
- `GET /health` - Detailed service status

### Chat Endpoints
- `POST /chat` - General chat with RAG
- `POST /chat/specialist` - Chat with specialist agents
- `GET /specialists` - List available specialists

### Utilities
- `GET /collection/stats` - Database statistics
- `DELETE /collection` - Clear all data (use with caution!)

## > Specialist Agents

### Physics Expert ›
- Mechanics, forces, torque, momentum
- Energy and power systems
- Control theory and stability

### Programming Expert =»
- Python, C++, ROS programming
- Algorithms and data structures
- Software architecture

### Mathematics Expert =Ð
- Linear algebra and transformations
- Calculus and optimization
- Mathematical foundations

### Book Expert =Ú
- General AI tutor
- Textbook knowledge
- Auto-coordination to specialists

## =' Usage Examples

### 1. Ask about Physics
```
"What is torque and how is it applied in humanoid robotics?"
```

### 2. Request Code
```
"Write Python code for PID controller"
```

### 3. Mathematics Query
```
"Explain rotation matrices for robot transformations"
```

### 4. Book Knowledge
```
"What is covered in Chapter 4 of the Physical AI textbook?"
```

## <¯ Auto-Routing Logic

The system automatically routes queries based on keywords:

- **Physics keywords**: torque, force, energy, momentum, etc.
- **Programming keywords**: code, program, python, algorithm, etc.
- **Mathematics keywords**: matrix, equation, calculate, vector, etc.
- **Book keywords**: chapter, textbook, definition, according to the book, etc.

## = Environment Variables

| Variable | Description | Required |
|----------|-------------|----------|
| `OPENAI_API_KEY` | OpenAI API key | Yes |
| `QDRANT_URL` | Qdrant Cloud URL | Yes |
| `QDRANT_API_KEY` | Qdrant API key | Yes |
| `COLLECTION_NAME` | Database collection name | No |
| `EMBED_MODEL` | Embedding model | No |

## =€ Deployment

### Docker (Recommended)

1. **Create Dockerfile**:
   ```dockerfile
   FROM python:3.11-slim
   WORKDIR /app
   COPY requirements.txt .
   RUN pip install -r requirements.txt
   COPY . .
   EXPOSE 8000
   CMD ["python", "main.py", "--port", "8000"]
   ```

2. **Build and run**:
   ```bash
   docker build -t rag-chatbot .
   docker run -p 8000:8000 rag-chatbot
   ```

### Cloud Deployment

The backend can be deployed to:
- AWS ECS/Fargate
- Google Cloud Run
- Azure Container Instances
- Railway
- Render

## =Ê Monitoring

- Health endpoint: `GET /health`
- Database stats: `GET /collection/stats`
- Logs available for debugging

## =' Development

### Adding New Specialists

1. Create new agent in `create_specialist_agents()`
2. Add to `specialistMap` in ChatWidget
3. Update auto-routing keywords

### Modifying Prompts

Edit agent instructions in `main.py`:
- Physics specialist
- Programming specialist
- Mathematics specialist
- General tutor

## =Ý License

This project is for educational purposes. Please ensure you have proper licenses for all dependencies.

## > Support

For issues or questions:
1. Check the health endpoint: `GET /health`
2. Verify environment variables
3. Check browser console for errors
4. Ensure backend is running on port 8000

---

**Made with d for Physical AI & Humanoid Robotics education**