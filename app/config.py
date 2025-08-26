"""Config.py"""

import os

from dotenv import load_dotenv

load_dotenv()


class Settings:
    """Encapsulates settings for app"""

    PROJECT_NAME: str = "Docling API Service"
    VERSION: str = "0.1.0"

    # API settings
    API_HOST: str = os.getenv("API_HOST", "0.0.0.0")
    API_PORT: int = int(os.getenv("API_PORT", 8991))

    # Embedding service settings
    EMBEDDING_SERVICE_URL: str = os.getenv("EMBEDDING_SERVICE_URL", "")
    EMBEDDING_SERVICE_API_KEY: str = os.getenv("EMBEDDING_SERVICE_API_KEY", "")
    EMBEDDING_MODEL: str = os.getenv(
        "EMBEDDING_MODEL", "intfloat/multilingual-e5-large-instruct"
    )

    # Redis for job queue (optional, can use in-memory for simple cases)
    REDIS_URL: str = os.getenv("REDIS_URL", "redis://localhost:6379/0")

    # File processing settings
    MAX_FILE_SIZE: int = 50 * 1024 * 1024  # 50MB
    UPLOAD_DIR: str = os.getenv("UPLOAD_DIR", "./uploads")
    PROCESSED_DIR: str = os.getenv("PROCESSED_DIR", "./processed")


settings = Settings()
