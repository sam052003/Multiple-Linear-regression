"""
FastAPI Application Entry Point.
Compatible with Vercel Serverless Functions and standalone Uvicorn execution.
"""

import sys
import os

# Ensure project root is in sys.path
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.schemas import HealthResponse
from api.routers import datasets, analysis, predict

app = FastAPI(
    title="Multiple Linear Regression Dashboard API",
    description="Backend API for CRISP-DM Multiple Linear Regression, 5 Feature Selection methods, 95% Prediction Interval, and Diagnostics.",
    version="2.0.0",
    docs_url="/api/docs",
    openapi_url="/api/openapi.json",
)

# CORS configuration for Next.js frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount Routers under /api
app.include_router(datasets.router, prefix="/api")
app.include_router(analysis.router, prefix="/api")
app.include_router(predict.router, prefix="/api")


@app.get("/api/health", response_model=HealthResponse, tags=["health"])
def health_check():
    """Health check endpoint to verify backend status."""
    return HealthResponse()


@app.get("/", tags=["root"])
def root_redirect():
    """Root redirect."""
    return {"message": "Multiple Linear Regression API", "docs": "/api/docs"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.index:app", host="0.0.0.0", port=8000, reload=True)
