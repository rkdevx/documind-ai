import os
from dotenv import load_dotenv

load_dotenv()

DATABASE_URL = os.getenv(
    "DATABASE_URL", 
    "postgresql://postgres:root@localhost:5432/documind_db"
)
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")