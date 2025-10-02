"""Configuration settings for the application."""
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Google Drive API
SCOPES = ['https://www.googleapis.com/auth/drive.readonly']

# Qdrant
COLLECTION_NAME = "drive_docs"
QDRANT_URL = os.getenv("QDRANT_URL", ":memory:")
VECTOR_SIZE = int(os.getenv("VECTOR_SIZE", "768"))
DISTANCE_METRIC = os.getenv("DISTANCE_METRIC", "COSINE")

# Application
TOP_K_RESULTS = int(os.getenv("TOP_K_RESULTS", "3"))
MAX_CONTENT_LENGTH = 10000  # Max characters to process per document

# Paths
BASE_DIR = Path(__file__).parent
CREDENTIALS_FILE = BASE_DIR / "credentials.json"

# Logging
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
