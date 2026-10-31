from fastapi import APIRouter, File, UploadFile, HTTPException
from app.tasks import process_document_task
from app.services.rag_service import extract_text_from_pdf

router = APIRouter(prefix="/documents", tags=["Documents"])

@router.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    """
    Accept .txt, .md, and .pdf files asynchronously and offload to Celery worker.
    """
    allowed_extensions = (".txt", ".md", ".pdf")
    if not file.filename.endswith(allowed_extensions):
        raise HTTPException(
            status_code=400, 
            detail="Only .txt, .md, and .pdf files are supported for ingestion."
        )
    
    try:
        content = await file.read()
        
        # Agar PDF hai toh pypdf se text extract karo, warna direct decode karo
        if file.filename.endswith(".pdf"):
            text_content = extract_text_from_pdf(content)
        else:
            text_content = content.decode("utf-8")
            
        if not text_content.strip():
            raise HTTPException(status_code=400, detail="The uploaded document is empty or unreadable.")
        
        # Trigger Celery Background Task
        task = process_document_task.delay(filename=file.filename, text_content=text_content)
        
        return {
            "message": "File uploaded successfully and queued for background processing!",
            "task_id": task.id,
            "filename": file.filename
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing file: {str(e)}")