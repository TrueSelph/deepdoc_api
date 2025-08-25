"""Models for the chunking service API."""

from enum import Enum
from typing import List, Optional, Dict

from pydantic import BaseModel, Field


class JobStatus(str, Enum):
    """Enumeration of possible job statuses."""

    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class ChunkMetadata(BaseModel):
    """Metadata associated with a text chunk."""

    page_num_int: List[int] = Field(default_factory=list)
    original_filename: str
    bbox: Optional[Dict[str, float]] = None
    chunk_size: Optional[int] = None
    chunk_overlap: Optional[int] = None


class ChunkResult(BaseModel):
    """Result of text chunking operation."""

    id: str
    metadata: ChunkMetadata
    text: str
    embeddings: Optional[List[float]] = None


class JobStatusResponse(BaseModel):
    """Response containing job status and results."""

    status: JobStatus
    result: Optional[List[ChunkResult]] = None
    error: Optional[str] = None


class UploadChunkRequest(BaseModel):
    """Request parameters for uploading and chunking documents."""

    agent_id: Optional[str] = None
    from_page: int = 0
    to_page: int = 100000
    lang: str = "english"
    with_embeddings: bool = False
    urls: Optional[List[str]] = None
    metadatas: Optional[List[dict]] = None
