# middleware.py

from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI

def configure_middleware(app: FastAPI):
    """
    Configure global middleware for FastAPI.
    Currently enabled:
      - CORS: Allow all origins, headers, and methods
      - (Extend here later if needed)
    """

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    return app
