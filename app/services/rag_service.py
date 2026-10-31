import numpy as np
from sentence_transformers import SentenceTransformer
from sqlalchemy.orm import Session
from app.models import Document, DocumentChunk
from sqlalchemy import select

import io
from pypdf import PdfReader

# Model ko memory mein ek hi baar load karenge taaki baar-baar load na karna pade
print("Loading embedding model (all-MiniLM-L6-v2)...")
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

def simple_text_splitter(text: str, chunk_size: int = 500, overlap: int = 50):
    """
    Simple & robust text chunker with overlap to keep context intact.
    """
    chunks = []
    start = 0
    text_length = len(text)
    
    while start < text_length:
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk.strip())
        start += chunk_size - overlap  # Overlap shift
        
    return [c for c in chunks if c] # Empty chunks remove karna

def ingest_document(db: Session, filename: str, raw_text: str):
    """
    1. Document record create karega
    2. Text ko chunks mein todega
    3. Vectors generate karke Database (`pgvector`) mein save karega
    """
    # 1. Parent document entry
    db_doc = Document(filename=filename)
    db.add(db_doc)
    db.commit()
    db.refresh(db_doc)
    
    # 2. Split text into chunks
    text_chunks = simple_text_splitter(raw_text, chunk_size=400, overlap=50)
    
    # 3. Generate embeddings in bulk for efficiency
    embeddings = embedding_model.encode(text_chunks, show_progress_bar=False)
    
    # 4. Save chunks and vectors to database
    for index, (chunk_text, embedding) in enumerate(zip(text_chunks, embeddings)):
        db_chunk = DocumentChunk(
            document_id=db_doc.id,
            chunk_index=index,
            content=chunk_text,
            embedding=embedding.tolist()  # pgvector expects list/array
        )
        db.add(db_chunk)
    
    db.commit()
    return {"status": "success", "document_id": db_doc.id, "total_chunks": len(text_chunks)}




def search_documents(
    db: Session,
    query_text: str,
    top_k: int = 3
) -> list[DocumentChunk]:
    """
    1. Generate query embedding.
    2. Perform cosine similarity search using pgvector.
    3. Return top-k relevant document chunks.
    """

    # Step 1: Generate query embedding
    query_embedding = embedding_model.encode(query_text).tolist()

    # Step 2: Build SQLAlchemy 2.0 select statement
    statement = (
        select(DocumentChunk)
        .order_by(
            DocumentChunk.embedding.cosine_distance(query_embedding)
        )
        .limit(top_k)
    )

    # Step 3: Execute query
    results = db.execute(statement).scalars().all()

    return list(results)



def extract_text_from_pdf(file_bytes: bytes) -> str:
    """PDF file bytes se saara text extract karega page-by-page"""
    reader = PdfReader(io.BytesIO(file_bytes))
    text = ""
    for page in reader.pages:
        extracted = page.extract_text()
        if extracted:
            text += extracted + "\n"
    return text
