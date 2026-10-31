from sqlalchemy import create_engine, text

from sqlalchemy.orm import sessionmaker
from app.config import DATABASE_URL
from app.base import Base

# SQLAlchemy engine create kar rahe hain
engine = create_engine(DATABASE_URL)

# Session local session factory
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def init_db():
    """
    It will enable pgvector extension 
    and create tables if not exists.
    """
    with engine.begin() as conn:
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector;"))
    print("PostgreSQL connection established and 'vector' extension enabled successfully!")

def get_db():
    """Dependency for FastAPI endpoints to get DB session"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


 
def create_tables():
    """Tables ko database mein create karega agar wo nahi bani hain"""
    Base.metadata.create_all(bind=engine)
    print("Database tables created successfully!")        