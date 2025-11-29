"""
main.py

FastAPI server for the chat backend.

Endpoints:
 - POST /refresh   -> clears both memories (list and chat); returns {"reset": True}
 - POST /query     -> body: {"query": str, "first_query": bool (optional)}
                     -> calls graph_func.examine_query(query, first_query)
                     -> returns {"response": str, "continue": bool}

Run:
    python main.py
"""

import os
import logging
from typing import Optional
from dotenv import load_dotenv

from fastapi import FastAPI, HTTPException
import uvicorn
from pydantic import BaseModel

import utils
import graph_func
import middleware

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("chat-backend")

app = FastAPI(title="Chat Backend", version="0.1.0")

# Apply middleware
middleware.configure_middleware(app)

@app.on_event("startup")
def startup_event():
    load_dotenv()
    # print("GROQ_API_KEY loaded:", os.getenv("GROQ_API_KEY"))

class QueryIn(BaseModel):
    query: str
    first_query: Optional[bool] = True


@app.post("/refresh")
def refresh():
    """
    Reset all memories.
    This endpoint performs a full reset of both 'list' and 'chat' memories.
    """
    try:
        utils.process_memory(form="list", type="update", content=[])
        utils.process_memory(form="chat", type="update", content=[])
        logger.info("Memories reset by /refresh")
        return {"reset": True}
    except Exception as e:
        logger.exception("Failed to reset memories")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/query")
def query_endpoint(payload: QueryIn):
    """
    Accepts a JSON payload with 'query' and optional 'first_query' (bool).
    Calls graph_func.examine_query(query, first_query) which should return (str, bool).
    Returns JSON: {"response": "<string>", "continue": <bool>}
    """
    q = payload.query
    first = bool(payload.first_query)

    if not isinstance(q, str) or q.strip() == "":
        raise HTTPException(status_code=400, detail="`query` must be a non-empty string.")

    try:
        response, continue_flag = graph_func.examine_query(query=q, first_query=first)

        if not (isinstance(response, str) and isinstance(continue_flag, bool)):
            raise ValueError("graph_func.examine_query must return (str, bool)")

        return {"response": response, "continue": continue_flag}
    except Exception as e:
        logger.exception("Error in /query")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)