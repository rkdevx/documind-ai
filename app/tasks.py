from app.celery_app import celery_app
from app.database import SessionLocal
from app.services.rag_service import ingest_document

@celery_app.task(bind=True, name="tasks.process_document_task")
def process_document_task(self, filename: str, text_content: str):
    """
    Celery background worker function jo heavy text chunking aur 
    vector embedding generation ko background mein run karega.
    """
    db = SessionLocal()
    try:
        print(f"Background worker started processing: {filename}")
        result = ingest_document(db=db, filename=filename, raw_text=text_content)
        print(f"Successfully processed and stored: {filename}")
        return result
    except Exception as e:
        print(f"Error in background task for {filename}: {str(e)}")
        raise e
    finally:
        db.close()