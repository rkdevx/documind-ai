
from pathlib import Path

# Project root
root = Path(".")

# Folder structure
folders = [
    "app",
    "app/routers",
    "app/services",
]

# Files to create
files = [
    "app/__init__.py",
    "app/main.py",
    "app/config.py",
    "app/database.py",
    "app/routers/__init__.py",
    "app/routers/upload.py",
    "app/routers/query.py",
    "app/services/__init__.py",
    "app/services/cache_service.py",
    "app/services/rag_service.py",
    "app/services/db_service.py",
    ".env",
    "README.md",
]

# Create folders
for folder in folders:
    (root / folder).mkdir(parents=True, exist_ok=True)

# Create files
for file in files:
    path = root / file
    path.touch(exist_ok=True)

print("DocuMind-AI project structure created successfully!")
