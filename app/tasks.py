"""Celery tasks for DeepDoc API"""

import asyncio
import logging
import os
import threading
from typing import Dict, List

from app.celery import celery_app
from app.models import JobStatus
from app.processing import document_processor
from celery.contrib.abortable import AbortableTask


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global storage for jobs
# In a production environment, consider using Redis or a database
jobs: Dict[str, Dict] = {}


@celery_app.task(bind=True, name="process_document", base=AbortableTask)
def process_document_task(
    self, job_id: str, file_paths: List[str], params: dict
) -> Dict:
    """
    Celery task to process documents

    Args:
        job_id: Unique identifier for the job
        file_paths: List of paths to files to process
        params: Additional parameters for processing

    Returns:
        Dict with job status, result, and error information
    """
    # Update job status to processing
    jobs[job_id] = {"status": JobStatus.PROCESSING, "result": None, "error": ""}

    try:
        all_chunks = []
        processed_files = 0

        for file_path in file_paths:
            # Check for task revocation using supported method
            if self.is_aborted():  # Correct method for checking revocation
                logger.info(f"Job {job_id} was cancelled during processing")
                jobs[job_id] = {
                    "status": JobStatus.CANCELLED,
                    "result": None,
                    "error": "Job was cancelled by user",
                }
                return jobs[job_id]

            try:
                logger.info(f"Processing file: {file_path}")

                # Process document
                chunks = document_processor.process_document(file_path, params)
                all_chunks.extend(chunks)
                processed_files += 1

                # Clean up processed file
                try:
                    os.remove(file_path)
                    logger.debug(f"Removed processed file: {file_path}")
                except OSError as e:
                    logger.warning(f"Could not remove file {file_path}: {e}")

            except Exception as e:
                logger.error(f"Failed to process file {file_path}: {e}")
                # Continue with other files even if one fails
                continue

        if processed_files == 0:
            raise Exception("No files were successfully processed")

        # Check for task revocation one final time
        if self.is_aborted():  # Correct method for checking revocation
            logger.info(f"Job {job_id} was cancelled after processing")
            jobs[job_id] = {
                "status": JobStatus.CANCELLED,
                "result": None,
                "error": "Job was cancelled by user after processing",
            }
            return jobs[job_id]

        # Update job status
        jobs[job_id] = {
            "status": JobStatus.COMPLETED,
            "result": [chunk.dict() for chunk in all_chunks],
            "error": "",
        }

        return jobs[job_id]

    except Exception as e:
        logger.error(f"Job {job_id} failed: {e}")
        jobs[job_id] = {"status": JobStatus.FAILED, "result": None, "error": str(e)}
        return jobs[job_id]


@celery_app.task(bind=True, name="trigger_callback")
def trigger_callback_task(_self, job_id: str, callback_url: str) -> None:
    """
    Task to trigger callback to notify about job completion

    Args:
        job_id: Unique identifier for the job
        callback_url: URL to send callback to
    """
    from app.main import _retry_callback_with_simple_payload

    # Run the async function in a new event loop
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    try:
        loop.run_until_complete(
            _retry_callback_with_simple_payload(job_id, callback_url)
        )
    finally:
        loop.close()
