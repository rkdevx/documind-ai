
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.database import init_db, create_tables
from app.routers import upload, query

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Application startup
    print("Starting DocuMind AI Engine...")

    init_db()
    create_tables()

    yield

    # Application shutdown
    print("Shutting down DocuMind AI Engine...")


app = FastAPI(
    title="DocuMind AI Engine",
    description="Production-grade Enterprise RAG & Semantic Caching Backend",
    version="1.0.0",
    lifespan=lifespan
)


# Register routers
app.include_router(upload.router)
app.include_router(query.router)


@app.get("/", tags=["Health"])
async def health_check():
    return {
        "status": "healthy",
        "message": "DocuMind AI Backend is up and running smoothly! 🚀"
    }
