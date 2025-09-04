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
from typing import Any, AsyncIterator, Dict, List, Tuple
from urllib.parse import unquote, urlparse

import aiofiles  # type: ignore
import requests
from fastapi import BackgroundTasks, FastAPI, HTTPException, UploadFile
from fastapi.params import Body
from celery.result import AsyncResult

from app.config import settings
from app.models import (
    ChunkMetadata,
    ChunkResult,
    JobStatus,
    JobStatusResponse,
)
from app.processing import document_processor
from app.tasks import process_document_task, trigger_callback_task

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variables for shared resources

executor = None

# these global variables are for job cancellation tracking
cancellation_events: Dict[str, threading.Event] = {}
processing_tasks: Dict[str, asyncio.Task] = {}


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Lifespan context manager for application startup and shutdown events."""
    global executor

    # Startup logic
    logger.info("Starting application...")

    # Create necessary directories
    os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
    os.makedirs(settings.PROCESSED_DIR, exist_ok=True)

    # Initialize thread pool executor for non-Celery tasks
    executor = ThreadPoolExecutor(max_workers=4)
    logger.info(
        f"Thread pool executor initialized with {executor._max_workers} workers"
    )

    logger.info("Application startup complete")

    # Yield control to the application
    yield

    # Shutdown logic
    logger.info("Shutting down application...")

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
        logger.error(f"Failed to download file from URL {url}: {e}")
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
        logger.error(f"Google Drive download failed: {e}")
        raise


async def save_upload_file(upload_file: UploadFile, destination: str) -> str:
    """Save uploaded file to destination and return the original filename"""
    try:
        async with aiofiles.open(destination, "wb") as f:
            while content := await upload_file.read(1024 * 1024):  # 1MB chunks
                await f.write(content)
    except Exception as e:
        logger.error(f"Error saving file: {e}")
        raise
    finally:
        await upload_file.close()
    return upload_file.filename  # Return the original filename, not the saved path


async def trigger_callback(job_id: str, callback_url: str) -> None:
    """Send job result to callback URL with proper error handling"""
    try:
        import requests

        task_result = AsyncResult(job_id)
        status = CELERY_TO_JOB_STATUS.get(task_result.status, JobStatus.FAILED)
        result = None
        error = None

        if task_result.successful():
            task_output = task_result.result
            if isinstance(task_output, dict):
                status = JobStatus(task_output.get("status", JobStatus.COMPLETED))
                result = task_output.get("result")
                error = task_output.get("error")
            else:
                status = JobStatus.COMPLETED
                result = task_result.result
        elif task_result.failed():
            status = JobStatus.FAILED
            error = str(task_result.info or task_result.result)
        elif task_result.status == "REVOKED":
            status = JobStatus.CANCELLED
            error = "Job was cancelled by user"

        # Prepare callback payload - ensure error is always a string, not null
        payload = {
            "job_id": job_id,
            "status": status,
            "result": result,
            "error": error or "",  # Convert null to empty string
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
            callback_url, json=payload, headers=headers, timeout=15
        )

        # Log the response for debugging
        logger.info(f"Callback response status: {response.status_code}")
        if response.status_code >= 400:
            logger.warning(f"Callback response content: {response.text}")

        response.raise_for_status()
        logger.info(f"Callback to {callback_url} successful for job {job_id}")

    except requests.exceptions.Timeout:
        logger.error(f"Callback timeout for job {job_id} to {callback_url}")
    except requests.exceptions.ConnectionError:
        logger.error(
            f"Connection error for callback to {callback_url} for job {job_id}"
        )
    except requests.exceptions.HTTPError as e:
        logger.error(
            f"HTTP error in callback for job {job_id}: {e.response.status_code} - {e.response.text}"
        )

        # If it's a 422 error, try with a different payload structure
        if e.response.status_code == 422:
            await _retry_callback_with_simple_payload(job_id, callback_url, payload)
    except Exception as e:
        logger.error(f"Callback failed for job {job_id}: {e}")


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
        logger.error(f"Retry callback also failed for job {job_id}: {retry_e}")


@app.post("/upload_and_chunk")
async def upload_and_chunk_endpoint(
    files: List[UploadFile] | None = None,
    urls: List[str] | None = Body(None),  # noqa: B008
    from_page: int = Body(0),  # noqa: B008
    to_page: int = Body(100000),  # noqa: B008
    lang: str = Body("english"),  # noqa: B008
    with_embeddings: bool = Body(False),  # noqa: B008
    callback_url: str | None = Body(None),  # noqa: B008
) -> Dict[str, str]:
    """Endpoint to process files asynchronously from uploads or URLs"""
    # Generate a temporary ID for file handling
    temp_id = str(uuid.uuid4())

    # Validate that we have at least one file source
    if not files and not urls:
        raise HTTPException(status_code=400, detail="No files or URLs provided")

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
                    url, temp_dir, temp_id
                )
                file_paths.append(file_path)
                original_filenames[file_path] = original_filename
                logger.info(f"Downloaded file from URL: {url} -> {file_path}")
            except Exception as e:
                logger.error(f"Failed to process URL {url}: {e}")
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
            file_path = os.path.join(settings.UPLOAD_DIR, f"{temp_id}_{file.filename}")
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            original_filename = await save_upload_file(file, file_path)
            file_paths.append(file_path)
            original_filenames[file_path] = original_filename  # Store original filename

    # Check if we have any files to process
    if not file_paths:
        raise HTTPException(status_code=400, detail="No valid files to process")

    # Prepare processing parameters
    params = {
        "from_page": from_page,
        "to_page": to_page,
        "lang": lang,
        "with_embeddings": with_embeddings,
        "callback_url": callback_url,
        "original_filenames": original_filenames,  # Pass the filename mapping
    }

    # Start Celery task for document processing
    if callback_url:
        # Chain the callback task to run after processing completes
        task = process_document_task.apply_async(
            args=[file_paths, params],
            link=trigger_callback_task.s(callback_url),
        )
    else:
        # Just run the processing task without a callback
        task = process_document_task.delay(file_paths, params)

    return {"job_id": task.id}


# Mapping from Celery states to JobStatus
CELERY_TO_JOB_STATUS = {
    "PENDING": JobStatus.PENDING,
    "STARTED": JobStatus.PROCESSING,
    "SUCCESS": JobStatus.COMPLETED,
    "FAILURE": JobStatus.FAILED,
    "REVOKED": JobStatus.CANCELLED,
    "RETRY": JobStatus.PROCESSING,
}


@app.get("/job/{job_id}")
async def get_job_status_endpoint(job_id: str) -> JobStatusResponse:
    """Endpoint to get the status of a job from the Celery backend."""
    task_result = AsyncResult(job_id)
    status = CELERY_TO_JOB_STATUS.get(task_result.status, JobStatus.FAILED)
    result = None
    error = None

    if task_result.successful():
        task_output = task_result.result
        if isinstance(task_output, dict):
            status = JobStatus(task_output.get("status", JobStatus.COMPLETED))
            result = task_output.get("result")
            error = task_output.get("error")
        else:
            status = JobStatus.COMPLETED
            result = task_result.result
    elif task_result.failed():
        status = JobStatus.FAILED
        error = str(task_result.info or task_result.result)
    elif task_result.status == "REVOKED":
        status = JobStatus.CANCELLED
        error = "Job was cancelled by user"

    return JobStatusResponse(
        job_id=job_id,
        status=status,
        result=result,
        error=error,
        created_at=(
            task_result.date_done.isoformat()
            if task_result.date_done
            else datetime.datetime.now(datetime.timezone.utc).isoformat()
        ),
    )


# Add the cancel endpoint
@app.post("/job/{job_id}/cancel")
async def cancel_job(job_id: str) -> Dict[str, str]:
    """Endpoint to cancel a job"""
    task_result = AsyncResult(job_id)

    if task_result.status in [
        JobStatus.COMPLETED,
        JobStatus.FAILED,
        JobStatus.CANCELLED,
    ]:
        return {"status": f"Job {job_id} is already {task_result.status}"}

    task_result.revoke(terminate=True)

    return {"status": f"Job {job_id} has been cancelled"}


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
        logger.error(f"Error cleaning up files for job {job_id}: {e}")


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

    return {
        "status": "healthy",
        "embedding_service": embedding_status,
        "executor": executor_status,
    }


@app.get("/liveness")
async def liveness_check() -> Dict[str, str]:
    """Lightweight health check endpoint"""
    return {"status": "alive"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host=settings.API_HOST, port=settings.API_PORT)
