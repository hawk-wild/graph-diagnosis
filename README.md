# Chat Application – Backend & Client Guide

This repository contains two main components:

- **server/** – FastAPI backend handling chat logic, memory management, and query evaluation.  
- **client/** – A simple frontend (or integration layer) that calls the backend endpoints.

This document explains the project structure, how the backend works, how to configure it, and how to run it locally on Windows, Linux, and macOS.

---

## 📁 Folder Structure

### **client/**
This folder contains your frontend or consumer code that talks to the backend API.  
Use the endpoints listed below to integrate your UI or agent workflow.

---

### **server/**
Contains the entire FastAPI backend.

#### **main.py**
Defines and runs the server.  
Exposes two endpoints:

- `POST /refresh` — resets both memory stores (`list` memory and `chat` memory).  
- `POST /query` — receives a query string and optionally `first_query`, then pipes it through `graph_func.examine_query()`.

It also loads environment variables, initializes middleware, and starts the FastAPI app.

#### **middleware.py**
Configures all middleware for the server.
- CORS configuration (allowed origins, headers, methods, credentials).
- Any additional request/response middleware your app uses.

#### **utils.py**
A utility layer simplifying backend operations such as:
- Updating or appending to memory.
- Fetching memory values.
- Abstracted LLM invocation wrappers if required.

Two memory “sites” are available:
1. **List memory** → used for storing structured info (like accumulating symptoms for a diagnosis).  
2. **Chat memory** → stores full conversational history.

Use the provided functions directly:
- `process_memory(form="list" or "chat", type="update"/"append"/"fetch", content=...)`

#### **graph_func.py**
This is where all query-processing logic goes.

You must implement:

```python
def examine_query(query: str, first_query: bool = True) -> tuple[str, bool]:
    """
    Returns:
        response: string to send back to client
        continue_flag: bool indicating whether backend expects more input
    """
```

Use:
- `process_memory` from utils
- both memory stores
- LLM calls provided in utils.py

This file contains the core behavior of the chat agent.

## 📌 API Endpoints (Client Reference)
### POST /refresh

Description:
Clears both memories (list and chat).

Response:
```json
{ "reset": true }
```

Usage Example:
```bash
POST http://localhost:8000/refresh
```

### POST /query

Body:
```json
{
  "query": "user message here",
  "first_query": true
}
```

Description:
Passes the user query into graph_func.examine_query().

Returns:
```json
{
  "response": "<string>",
  "continue": true or false
}
```

Usage Example:
```bash
POST http://localhost:8000/query
```

## ⚙️ Backend Setup
1. Navigate to the backend
```bash
cd server
```
2. Create virtual environment
Windows
```bash
python -m venv venv
venv\Scripts\activate
```
Linux / macOS
```bash
python3 -m venv venv
source venv/bin/activate
```
3. Install dependencies
```bash
pip install -r requirements.txt
```
🔐 Environment Variables

Inside server/, copy the example environment file:

Linux / macOS:
```bash
cp .env.example .env
```

Windows PowerShell:
```powershell
copy .env.example .env
```

Open the new .env file and replace the placeholder:

```env
GROQ_API_KEY=your_api_key_here
```

▶️ Running the Backend
Windows
```bash
python main.py
```
Linux / macOS
```bash
python3 main.py
```

This launches FastAPI at:
```bash
http://127.0.0.1:8000
```

✔️ Additional Notes
- .env must exist before starting the server.
- `examine_query()` must return exactly (response_string, continue_flag)
- Use process_memory for all memory operations.
- Middleware is applied automatically via `middleware.configure_middleware(app)`.