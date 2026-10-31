from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session
from app.database import get_db
from app.services.rag_service import search_documents, embedding_model
from app.services.cache_service import get_semantic_cached_answer, set_semantic_cache
from app.services.llm_service import generate_rag_answer

router = APIRouter(prefix="/query", tags=["RAG Query"])

class QueryRequest(BaseModel):
    question: str
    top_k: int = 3

@router.post("/")
async def ask_question(payload: QueryRequest, db: Session = Depends(get_db)):
    try:
        # 1. Query embedding generate karo
        query_embedding = embedding_model.encode(payload.question).tolist()
        
        # 2. Check Redis Semantic Cache first (< 0.05s response if hit)
        cached_result = await get_semantic_cached_answer(query_embedding)
        if cached_result:
            return {
                "question": payload.question,
                **cached_result
            }
        
        # 3. Cache Miss - PostgreSQL pgvector se relevant chunks search karo
        chunks = search_documents(db, payload.question, payload.top_k)
        
        if not chunks:
            return {
                "question": payload.question,
                "cached": False,
                "answer": "No relevant context found in the uploaded documents.",
                "sources": []
            }
        
        # 4. Context combine karo
        context = "\n\n".join([c.content for c in chunks])
        sources = list(set([c.document.filename for c in chunks]))
        
        # 5. Local PyTorch LLM (Phi-3-mini) ke through final answer generate karo!
        print("Generating answer using local GPU LLM...")
        llm_answer = generate_rag_answer(context=context, question=payload.question)
        
        response_data = {
            "cached": False,
            "answer": llm_answer,
            "sources": sources
        }
        
        # 6. Future ke liye Redis semantic cache mein save kar do
        await set_semantic_cache(payload.question, query_embedding, llm_answer, sources)
        
        return {
            "question": payload.question,
            **response_data
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during query processing: {str(e)}")