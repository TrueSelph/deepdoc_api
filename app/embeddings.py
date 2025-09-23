"""Embedding client for generating embeddings using JivasEmbeddings service"""

import logging
from typing import List, Optional

from app.config import settings
from app.jivas_embeddings import JivasEmbeddings

logger = logging.getLogger(__name__)


# In app/embeddings.py
class EmbeddingClient:
    """Embedding client for generating embeddings using JivasEmbeddings service"""

    def __init__(self) -> None:
        """Initialize the EmbeddingClient with JivasEmbeddings service"""
        self.url = settings.EMBEDDING_SERVICE_URL
        self.api_key = settings.EMBEDDING_SERVICE_API_KEY
        self.model = (
            settings.EMBEDDING_MODEL
            if hasattr(settings, "EMBEDDING_MODEL")
            else "intfloat/multilingual-e5-large-instruct"
        )
        self.client = None

        # Initialize the JivasEmbeddings client if configured
        if self.url and self.api_key:
            try:
                self.client = JivasEmbeddings(
                    base_url=self.url, api_key=self.api_key, model=self.model
                )
                logger.info("JivasEmbeddings client initialized successfully")
            except Exception as e:
                logger.exception(f"Failed to initialize JivasEmbeddings: {e}")
                self.client = None
        else:
            logger.warning("Embedding service not configured (missing URL or API key)")

    def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a list of texts using the JivasEmbeddings service"""
        if not self.client:
            logger.error("Embedding service not configured")
            raise ValueError("Embedding service not configured")

        try:
            # Use the embed_documents method with overflow handling
            embeddings = self.client.embed_documents(texts, handle_overflow=True)
            return embeddings
        except Exception as e:
            logger.exception(f"Error generating embeddings: {e}")
            raise

    def generate_embedding(self, text: str) -> Optional[List[float]]:
        """Generate embedding for a single text using the JivasEmbeddings service"""
        if not self.client:
            logger.error("Embedding service not configured")
            return None

        try:
            # Use the embed_query method for single text
            embedding = self.client.embed_query(text, handle_overflow=True)
            return embedding
        except Exception as e:
            logger.exception(f"Error generating embedding for text: {e}")
            return None


# Global embedding client instance
embedding_client = EmbeddingClient()
