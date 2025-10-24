"""Entrypoint for the DeepDoc API"""

import asyncio
import datetime
import json
import logging
import mimetypes
import os
import re
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator, Dict, List, Optional, Tuple
from urllib.parse import unquote, urlparse

import aiofiles  # type: ignore
import redis
import requests
from fastapi import BackgroundTasks, FastAPI, HTTPException, UploadFile
from fastapi.params import Body

from app.config import settings
from app.models import (
    ChunkMetadata,
    ChunkResult,
    JobStatus,
    JobStatusResponse,
)
from app.processing import document_processor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variables for shared resources
redis_client = None
executor = None

# these global variables are for job cancellation tracking
cancellation_events: Dict[str, threading.Event] = {}
processing_tasks: Dict[str, asyncio.Task] = {}


def set_job_data(job_id: str, job_data: dict) -> None:
    """Store job data in Redis"""
    if redis_client:
        job_key = f"job:{job_id}"
        # Convert None to empty string for error field
        if job_data.get("error") is None:
            job_data["error"] = ""
        redis_client.set(job_key, json.dumps(job_data))
        # Set expiration to 24 hours
        redis_client.expire(job_key, 86400)


def get_job_data(job_id: str) -> dict | None:
    """Retrieve job data from Redis"""
    if redis_client:
        job_key = f"job:{job_id}"
        job_json = redis_client.get(job_key)
        if job_json:
            return json.loads(job_json)
    return None


def delete_job_data(job_id: str) -> None:
    """Delete job data from Redis"""
    if redis_client:
        job_key = f"job:{job_id}"
        redis_client.delete(job_key)


def update_job_progress(
    job_id: str,
    stage: str,
    progress: Optional[int] = None,
    details: Optional[Dict[str, Any]] = None,
) -> None:
    """Update job progress information in Redis"""
    if redis_client:
        job_data = get_job_data(job_id) or {}
        job_data["stage"] = stage
        if progress is not None:
            job_data["progress"] = progress
        if details:
            job_data["details"] = details
        set_job_data(job_id, job_data)


def log_progress_milestone(job_id: str, milestone: str, progress: int) -> None:
    """Log a progress milestone with timestamp"""
    timestamp = datetime.datetime.utcnow().isoformat() + "Z"
    logger.info(f"[{timestamp}] Job {job_id}: {milestone} ({progress}%)")
    update_job_progress(job_id, milestone, progress)


def get_job_counts() -> tuple[int, int, int]:
    """Get counts of jobs by status from Redis"""
    processing = 0
    pending = 0
    total = 0
    if redis_client:
        # Get all job keys
        job_keys = redis_client.keys("job:*")
        total = len(job_keys)
        for key in job_keys:
            job_data = get_job_data(key[4:])  # Remove "job:" prefix
            if job_data:
                if job_data.get("status") == JobStatus.PROCESSING:
                    processing += 1
                elif job_data.get("status") == JobStatus.PENDING:
                    pending += 1
    return processing, pending, total


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Lifespan context manager for application startup and shutdown events."""
    global executor

    # Startup logic
    logger.info("Starting application...")

    # Create necessary directories
    os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
    os.makedirs(settings.PROCESSED_DIR, exist_ok=True)

    # Initialize Redis client
    global redis_client
    redis_client = redis.from_url(settings.REDIS_URL, decode_responses=True)
    logger.info(f"Redis client initialized with URL: {settings.REDIS_URL}")

    # Initialize thread pool executor
    executor = ThreadPoolExecutor(max_workers=4)
    logger.info(
        f"Thread pool executor initialized with {executor._max_workers} workers"
    )

    logger.info("Application startup complete")

    # Yield control to the application
    yield

    # Shutdown logic
    logger.info("Shutting down application...")

    # Cancel all ongoing jobs
    # Note: Since jobs are now in Redis, we need to scan them
    if redis_client:
        job_keys = redis_client.keys("job:*")
        for key in job_keys:
            job_id = key[4:]  # Remove "job:" prefix
            job = get_job_data(job_id)
            if job and job.get("status") == JobStatus.PROCESSING:
                logger.info(f"Cancelling job {job_id} during shutdown")
                if job_id in cancellation_events:
                    cancellation_events[job_id].set()
                job["status"] = JobStatus.CANCELLED
                job["error"] = "Job was cancelled during server shutdown"
                set_job_data(job_id, job)

    # Shutdown thread pool executor
    if executor:
        executor.shutdown(wait=False)
        logger.info("Thread pool executor shutdown")

    logger.info("Application shutdown complete")


# Create FastAPI app with lifespan management
app = FastAPI(title=settings.PROJECT_NAME, version=settings.VERSION, lifespan=lifespan)


def is_google_drive_url(url: str) -> bool:
    """Check if a URL is a Google Drive URL"""
    return "drive.google.com" in url or "docs.google.com" in url


def get_filename_from_response(response: requests.Response, url: str) -> str:
    """Extract filename from response headers or URL"""
    # Try to get filename from Content-Disposition header
    content_disposition = response.headers.get("content-disposition", "")
    if content_disposition:
        fname = re.findall('filename="([^"]+)"', content_disposition)
        if fname:
            return unquote(fname[0])

    # Fallback to extracting from URL path
    parsed = urlparse(url)
    filename = unquote(os.path.basename(parsed.path))

    # Add extension if missing
    if not os.path.splitext(filename)[1]:
        content_type = response.headers.get("Content-Type", "")
        ext = mimetypes.guess_extension(content_type) or ".bin"
        filename = filename + ext

    return filename


async def download_file_from_url(
    url: str, output_dir: str, job_id: str
) -> Tuple[str, str]:
    """Download a file from a URL and return (file_path, original_filename)"""

    try:
        # Handle Google Drive URLs
        if is_google_drive_url(url):
            try:
                import gdown  # Import here to avoid dependency if not used
            except ImportError:
                raise ImportError(
                    "gdown package is required for Google Drive downloads. Install with: pip install gdown"
                )

            return await download_google_drive_file(url, output_dir, job_id, gdown)

        # Handle regular URLs
        else:
            response = requests.get(
                url,
                stream=True,
                headers={
                    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
                },
                timeout=30,
            )
            response.raise_for_status()

            # Get filename
            original_filename = get_filename_from_response(response, url)

            # Create a unique filename with job ID prefix
            unique_filename = f"{job_id}_{original_filename}"
            temp_file_path = os.path.join(output_dir, unique_filename)

            # Save the file
            with open(temp_file_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)

            return temp_file_path, original_filename

    except Exception as e:
        logger.exception(f"Failed to download file from URL {url}: {e}")
        raise HTTPException(
            status_code=400, detail=f"Failed to download file from URL: {str(e)}"
        )


async def download_google_drive_file(
    url: str,
    output_dir: str,
    job_id: str,
    gdown_module: Any,  # noqa: ANN401
) -> Tuple[str, str]:
    """Download a file from Google Drive using gdown"""
    try:
        # Extract file ID from Google Drive URL
        file_id_match = re.search(r"/d/([a-zA-Z0-9_-]+)", url) or re.search(
            r"id=([a-zA-Z0-9_-]+)", url
        )
        if not file_id_match:
            raise ValueError("Invalid Google Drive URL format")

        file_id = file_id_match.group(1)

        # Download using gdown
        temp_file_path = os.path.join(output_dir, f"{job_id}_gdrive_{file_id}")

        # Download the file
        gdown_module.download(
            f"https://drive.google.com/uc?id={file_id}", temp_file_path, quiet=False
        )

        # Try to get the original filename
        original_filename = file_id

        return temp_file_path, original_filename

    except Exception as e:
        logger.exception(f"Google Drive download failed: {e}")
        raise


async def save_upload_file(upload_file: UploadFile, destination: str) -> str:
    """Save uploaded file to destination and return the original filename"""
    try:
        async with aiofiles.open(destination, "wb") as f:
            while content := await upload_file.read(1024 * 1024):  # 1MB chunks
                await f.write(content)
    except Exception as e:
        logger.exception(f"Error saving file: {e}")
        raise
    finally:
        await upload_file.close()
    return upload_file.filename  # Return the original filename, not the saved path


async def process_job(job_id: str, file_paths: List[str], params: dict) -> None:
    """Background task to process documents for a job with cancellation support"""
    # Create cancellation event for this job
    cancellation_event = threading.Event()
    cancellation_events[job_id] = cancellation_event

    try:
        # Initialize job with progress tracking
        job_start_time = datetime.datetime.utcnow()
        initial_details = {
            "total_files": len(file_paths),
            "processed_files": 0,
            "current_operation": "Starting processing",
            "fallback_triggered": False,
            "start_time": job_start_time.isoformat() + "Z",
        }
        set_job_data(
            job_id,
            {
                "status": JobStatus.PROCESSING,
                "result": None,
                "error": "",
                "stage": "Initializing job",
                "progress": 0,
                "details": initial_details,
            },
        )

        log_progress_milestone(job_id, "Job processing initialized", 5)

        all_chunks = []
        processed_files = 0

        for file_path in file_paths:
            # Check for cancellation before processing each file
            if cancellation_event.is_set():
                logger.info(f"Job {job_id} was cancelled during processing")
                set_job_data(
                    job_id,
                    {
                        "status": JobStatus.CANCELLED,
                        "result": None,
                        "error": "Job was cancelled by user",
                    },
                )
                return

            try:
                file_name = os.path.basename(file_path)
                file_size = (
                    os.path.getsize(file_path) if os.path.exists(file_path) else 0
                )
                file_size_mb = file_size / (1024 * 1024)

                logger.info(
                    f"Processing file {processed_files + 1}/{len(file_paths)}: {file_name} ({file_size_mb:.1f} MB)"
                )

                # Update progress for current file
                progress = 10 + int((processed_files / len(file_paths)) * 70)
                update_job_progress(
                    job_id,
                    f"Processing file {processed_files + 1}/{len(file_paths)}: {file_name}",
                    progress,
                    {
                        "total_files": len(file_paths),
                        "processed_files": processed_files + 1,
                        "current_file": file_name,
                        "file_size_bytes": file_size,
                        "file_size_mb": round(file_size_mb, 1),
                        "current_operation": "Document processing",
                        "start_time": datetime.datetime.utcnow().isoformat() + "Z",
                    },
                )

                # Process document using thread pool (Docling is CPU-intensive)
                loop = asyncio.get_event_loop()
                chunks = await loop.run_in_executor(
                    executor,
                    document_processor.process_document,
                    file_path,
                    params,  # Pass the params which includes original_filenames
                )
                all_chunks.extend(chunks)
                processed_files += 1

                # Update progress after file completion
                progress = 10 + int((processed_files / len(file_paths)) * 70)
                update_job_progress(
                    job_id,
                    f"Completed file {processed_files}/{len(file_paths)}",
                    progress,
                    {
                        "total_files": len(file_paths),
                        "processed_files": processed_files,
                        "chunks_generated": len(all_chunks),
                    },
                )

                # Clean up processed file
                try:
                    os.remove(file_path)
                    logger.debug(f"Removed processed file: {file_path}")
                except OSError as e:
                    logger.warning(f"Could not remove file {file_path}: {e}")

            except Exception as e:
                logger.exception(f"Failed to process file {file_path}: {e}")
                # Continue with other files even if one fails
                continue

        if processed_files == 0:
            raise Exception("No files were successfully processed")

        # Check for cancellation one final time before marking as complete
        if cancellation_event.is_set():
            logger.info(f"Job {job_id} was cancelled after processing")
            set_job_data(
                job_id,
                {
                    "status": JobStatus.CANCELLED,
                    "result": None,
                    "error": "Job was cancelled by user after processing",
                },
            )
            return

        # Calculate processing statistics
        end_time = datetime.datetime.utcnow()
        total_processing_time = (end_time - job_start_time).total_seconds()

        # Calculate throughput metrics
        total_file_size = sum(
            os.path.getsize(fp) if os.path.exists(fp) else 0 for fp in file_paths
        )
        throughput_mbps = (
            (total_file_size / (1024 * 1024)) / max(total_processing_time / 3600, 0.001)
            if total_processing_time > 0
            else 0
        )

        # Update job status to completed
        log_progress_milestone(job_id, "Processing completed successfully", 95)
        set_job_data(
            job_id,
            {
                "status": JobStatus.COMPLETED,
                "result": [chunk.dict() for chunk in all_chunks],
                "error": "",  # Empty string instead of null
                "stage": "Completed",
                "progress": 100,
                "details": {
                    "total_files": len(file_paths),
                    "processed_files": processed_files,
                    "total_chunks": len(all_chunks),
                    "embeddings_generated": params.get("with_embeddings", False),
                    "total_file_size_bytes": total_file_size,
                    "total_file_size_mb": round(total_file_size / (1024 * 1024), 1),
                    "processing_time_seconds": round(total_processing_time, 1),
                    "throughput_mb_per_hour": round(throughput_mbps, 2),
                    "chunks_per_second": round(
                        len(all_chunks) / max(total_processing_time, 1), 2
                    ),
                },
            },
        )

        log_progress_milestone(
            job_id,
            f"Job completed - {len(all_chunks)} chunks, {round(total_processing_time, 1)}s, {round(throughput_mbps, 2)} MB/hr",
            100,
        )

        # Trigger callback if provided
        if params.get("callback_url"):
            await trigger_callback(job_id, params["callback_url"])

    except Exception as e:
        logger.exception(f"Job {job_id} failed: {e}")
        error_details = {
            "total_files": len(file_paths),
            "processed_files": processed_files,
            "error_type": type(e).__name__,
            "error_message": str(e),
            "failed_at_stage": "document_processing",
        }
        set_job_data(
            job_id,
            {
                "status": JobStatus.FAILED,
                "result": None,
                "error": str(e),
                "stage": "Failed",
                "progress": 0,
                "details": error_details,
            },
        )

        # Trigger callback for failure if provided
        if params.get("callback_url"):
            await trigger_callback(job_id, params["callback_url"])
    finally:
        # Clean up cancellation event
        if job_id in cancellation_events:
            del cancellation_events[job_id]
        if job_id in processing_tasks:
            del processing_tasks[job_id]


async def trigger_callback(job_id: str, callback_url: str) -> None:
    """Send job result to callback URL with proper error handling"""
    try:
        import requests

        job_data = get_job_data(job_id) or {}

        # Prepare callback payload - ensure error is always a string, not null
        payload = {
            "job_id": job_id,
            "status": job_data.get("status"),
            "result": job_data.get("result"),
            "error": job_data.get("error") or "",  # Convert null to empty string
            "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
        }

        # Add more debug info
        logger.info(f"Sending callback to {callback_url} for job {job_id}")
        logger.debug(f"Callback payload: {json.dumps(payload, indent=2)}")

        # Send POST request to callback URL with timeout and better headers
        headers = {
            "Content-Type": "application/json",
            "User-Agent": f"{settings.PROJECT_NAME}/{settings.VERSION}",
        }

        response = requests.post(
            callback_url, json=payload, headers=headers, timeout=1440
        )

        # Log the response for debugging
        logger.info(f"Callback response status: {response.status_code}")
        if response.status_code >= 400:
            logger.warning(f"Callback response content: {response.text}")

        response.raise_for_status()
        logger.info(f"Callback to {callback_url} successful for job {job_id}")

    except requests.exceptions.Timeout:
        logger.exception(f"Callback timeout for job {job_id} to {callback_url}")
    except requests.exceptions.ConnectionError:
        logger.exception(
            f"Connection error for callback to {callback_url} for job {job_id}"
        )
    except requests.exceptions.HTTPError as e:
        logger.exception(
            f"HTTP error in callback for job {job_id}: {e.response.status_code} - {e.response.text}"
        )

        # If it's a 422 error, try with a different payload structure
        if e.response.status_code == 422:
            await _retry_callback_with_simple_payload(job_id, callback_url, job_data)
    except Exception as e:
        logger.exception(f"Callback failed for job {job_id}: {e}")


async def _retry_callback_with_simple_payload(
    job_id: str, callback_url: str, job_data: dict
) -> None:
    """Retry callback with a simpler payload structure for compatibility"""
    try:
        import requests

        # Create a simpler payload that matches common webhook expectations
        simple_payload = {
            "job_id": job_id,
            "status": job_data.get("status"),
            "result": job_data.get("result"),
            "error": job_data.get("error") or "",  # Ensure error is never null
        }

        # Remove None values to avoid validation issues
        simple_payload = {k: v for k, v in simple_payload.items() if v is not None}

        logger.info(f"Retrying callback with simplified payload for job {job_id}")

        headers = {
            "Content-Type": "application/json",
            "User-Agent": f"{settings.PROJECT_NAME}/{settings.VERSION}",
        }

        response = requests.post(
            callback_url, json=simple_payload, headers=headers, timeout=10
        )

        response.raise_for_status()
        logger.info(f"Retry callback to {callback_url} successful for job {job_id}")

    except Exception as retry_e:
        logger.exception(f"Retry callback also failed for job {job_id}: {retry_e}")


@app.post("/upload_and_chunk")
async def upload_and_chunk_endpoint(
    background_tasks: BackgroundTasks,
    files: List[UploadFile] | None = None,
    urls: List[str] | None = Body(None),  # noqa: B008
    from_page: int = Body(0),  # noqa: B008
    to_page: int = Body(100000),  # noqa: B008
    lang: str = Body("english"),  # noqa: B008
    with_embeddings: bool = Body(False),  # noqa: B008
    callback_url: str | None = Body(None),  # noqa: B008
    chunker_type: str = Body("hybrid"),  # noqa: B008
) -> Dict[str, str]:
    logger.warning("##############")
    logger.warning(f"chunker_type: {chunker_type}")
    logger.warning("##############")
    """Endpoint to process files asynchronously from uploads or URLs"""
    # Generate job ID
    job_id = str(uuid.uuid4())

    # Validate that we have at least one file source
    if not files and not urls:
        raise HTTPException(status_code=400, detail="No files or URLs provided")

    logger.info("Starting file processing job %s", job_id)

    # Process URLs if provided
    file_paths = []
    original_filenames = {}  # Store mapping of file_path -> original_filename

    if urls:
        # Create temp directory if it doesn't exist
        temp_dir = os.path.join(settings.UPLOAD_DIR, "temp")
        os.makedirs(temp_dir, exist_ok=True)

        # Download files from URLs
        for url in urls:  # Directly iterate over the list
            try:
                file_path, original_filename = await download_file_from_url(
                    url, temp_dir, job_id
                )
                file_paths.append(file_path)
                original_filenames[file_path] = original_filename
                logger.info(f"Downloaded file from URL: {url} -> {file_path}")
            except Exception as e:
                logger.exception(f"Failed to process URL {url}: {e}")
                # Continue with other URLs instead of failing the entire request

    # Process uploaded files if provided
    if files:
        for file in files:
            # Validate file type
            file_ext = os.path.splitext(file.filename)[1].lower()
            if file_ext not in document_processor.supported_formats:
                raise HTTPException(
                    status_code=400, detail=f"Unsupported file format: {file_ext}"
                )

            # Save uploaded file with job ID prefix for uniqueness
            file_path = os.path.join(settings.UPLOAD_DIR, f"{job_id}_{file.filename}")
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            original_filename = await save_upload_file(file, file_path)
            file_paths.append(file_path)
            original_filenames[file_path] = original_filename  # Store original filename

    # Check if we have any files to process
    if not file_paths:
        raise HTTPException(status_code=400, detail="No valid files to process")

    logger.info("Job %s will process %d files", job_id, len(file_paths))

    # Prepare processing parameters
    params = {
        "from_page": from_page,
        "to_page": to_page,
        "lang": lang,
        "with_embeddings": with_embeddings,
        "callback_url": callback_url,
        "original_filenames": original_filenames,  # Pass the filename mapping
        "job_id": job_id,  # Pass job_id for progress tracking
        "chunker_type": chunker_type
    }

    if with_embeddings:
        logger.info("Job %s will generate embeddings", job_id)

    # Initialize job status
    set_job_data(job_id, {"status": JobStatus.PENDING, "result": None, "error": None})

    # Start background processing and track the task
    task = background_tasks.add_task(process_job, job_id, file_paths, params)
    processing_tasks[job_id] = task

    return {"job_id": job_id}


@app.get("/job/{job_id}")
async def get_job_status_endpoint(job_id: str) -> JobStatusResponse:
    """Endpoint to check the status of a job"""
    job = get_job_data(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    # Convert the stored result back to ChunkResult objects if needed
    result = job.get("result")
    if (
        result
        and isinstance(result, list)
        and len(result) > 0
        and isinstance(result[0], dict)
    ):
        # Convert dicts back to ChunkResult objects for proper serialization
        chunk_results = []
        for chunk_data in result:
            if isinstance(chunk_data, dict):
                # Handle both old and new format during transition
                if "metadata" in chunk_data and isinstance(
                    chunk_data["metadata"], dict
                ):
                    # New format with metadata object
                    chunk_results.append(ChunkResult(**chunk_data))
                else:
                    # Old format - convert to new format
                    chunk_id = chunk_data.get("id", f"chunk_{uuid.uuid4().hex}")
                    metadata = ChunkMetadata(
                        page_num_int=[1],  # Default page
                        original_filename=chunk_data.get("metadata", {}).get(
                            "original_filename", "unknown"
                        ),
                        chunk_size=chunk_data.get("metadata", {}).get("chunk_size"),
                        chunk_overlap=chunk_data.get("metadata", {}).get(
                            "chunk_overlap"
                        ),
                    )
                    chunk_results.append(
                        ChunkResult(
                            id=chunk_id,
                            metadata=metadata,
                            text=chunk_data.get("text", ""),
                            embedding=chunk_data.get("embedding"),
                        )
                    )
        job["result"] = chunk_results

    # Ensure error is never null - convert to empty string if None
    error = job.get("error")
    if error is None:
        error = ""

    # Extract progress information
    stage = job.get("stage")
    progress = job.get("progress")
    details = job.get("details")

    return JobStatusResponse(
        status=job["status"],
        result=job["result"],
        error=error,
        stage=stage,
        progress=progress,
        details=details,
    )


# Add the cancel endpoint
@app.post("/job/{job_id}/cancel")
async def cancel_job(job_id: str) -> Dict[str, object]:
    """Cancel a job - always returns success even if job doesn't exist or can't be cancelled"""
    # Check if job exists
    job = get_job_data(job_id)
    if not job:
        # Job doesn't exist, but still return success
        logger.warning(f"Cancel requested for non-existent job: {job_id}")
        return {
            "status": "cancelled",
            "message": f"Job {job_id} does not exist or was already completed",
            "job_id": job_id,
        }

    # Check if job can be cancelled
    if job["status"] in [JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED]:
        # Job is already in a terminal state, but still return success
        logger.info(
            f"Cancel requested for job {job_id} which is already in {job['status']} state"
        )
        return {
            "status": "cancelled",
            "message": f"Job {job_id} was already in {job['status']} state",
            "job_id": job_id,
            "previous_status": job["status"],
        }

    # If job is processing, attempt to cancel it
    if job["status"] == JobStatus.PROCESSING and job_id in cancellation_events:
        cancellation_events[job_id].set()
        logger.info(f"Cancellation requested for processing job {job_id}")

    # If job is pending, we can just mark it as cancelled
    elif job["status"] == JobStatus.PENDING:
        logger.info(f"Cancellation requested for pending job {job_id}")

    # Update job status to cancelled regardless of previous state
    job["status"] = JobStatus.CANCELLED
    job["error"] = "Job was cancelled by user"
    set_job_data(job_id, job)

    # Clean up any processing files for this job
    await cleanup_job_files(job_id)

    logger.info(f"Job {job_id} marked as cancelled")
    return {
        "status": "cancelled",
        "message": f"Job {job_id} has been cancelled",
        "job_id": job_id,
    }


async def cleanup_job_files(job_id: str) -> None:
    """Clean up files associated with a cancelled job"""
    try:
        # Find and delete any files that start with the job ID
        upload_dir = settings.UPLOAD_DIR
        if os.path.exists(upload_dir):
            for filename in os.listdir(upload_dir):
                if filename.startswith(job_id):
                    file_path = os.path.join(upload_dir, filename)
                    try:
                        os.remove(file_path)
                        logger.debug(f"Cleaned up file: {file_path}")
                    except OSError as e:
                        logger.warning(f"Could not remove file {file_path}: {e}")
    except Exception as e:
        logger.exception(f"Error cleaning up files for job {job_id}: {e}")


@app.get("/health")
async def health_check() -> Dict[str, object]:
    """Enhanced health check endpoint"""
    # Check if embedding service is available if configured
    if settings.EMBEDDING_SERVICE_URL:
        try:
            # Simple connection test
            import requests

            response = requests.get(settings.EMBEDDING_SERVICE_URL, timeout=5)
            embedding_status = (
                "connected" if response.status_code < 500 else "unavailable"
            )
        except Exception as e:
            embedding_status = f"error: {str(e)}"
    else:
        embedding_status = "not configured"

    # Check if executor is running
    executor_status = "running" if executor and not executor._shutdown else "stopped"

    # Get job counts
    jobs_processing, jobs_pending, total_jobs = get_job_counts()

    return {
        "status": "healthy",
        "embedding_service": embedding_status,
        "executor": executor_status,
        "jobs_processing": jobs_processing,
        "jobs_pending": jobs_pending,
        "total_jobs": total_jobs,
    }


@app.get("/liveness")
async def liveness_check() -> Dict[str, str]:
    """Lightweight health check endpoint"""
    return {"status": "alive"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host=settings.API_HOST, port=settings.API_PORT)
