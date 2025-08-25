from fastapi import UploadFile
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any, Union
from enum import Enum

class JobStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

class ChunkMetadata(BaseModel):
    page_num_int: List[int] = Field(default_factory=list)
    original_filename: str
    # Additional metadata fields can be added here
    chunk_size: Optional[int] = None
    chunk_overlap: Optional[int] = None

class ChunkResult(BaseModel):
    id: str
    metadata: ChunkMetadata
    text: str
    embeddings: Optional[List[float]] = None

class JobStatusResponse(BaseModel):
    status: JobStatus
    result: Optional[List[ChunkResult]] = None
    error: Optional[str] = None

class UploadChunkRequest(BaseModel):
    agent_id: Optional[str] = None
    from_page: int = 0
    to_page: int = 100000
    lang: str = "english"
    with_embeddings: bool = False
    urls: Optional[List[str]] = None
    metadatas: Optional[List[dict]] = None