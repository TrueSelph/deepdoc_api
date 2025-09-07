"""Celery tasks for DeepDoc API"""

import asyncio
import logging
import os
import threading
from typing import Dict, List

from app.celery_app import celery_app
from app.models import JobStatus
from app.processing import document_processor
from celery.contrib.abortable import AbortableTask


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global storage for jobs
# In a production environment, consider using Redis or a database


@celery_app.task(bind=True, name="process_document", base=AbortableTask)
def process_document_task(self, file_paths: List[str], params: dict) -> Dict:
    """
    Celery task to process documents

    Args:
        file_paths: List of paths to files to process
        params: Additional parameters for processing

    Returns:
        Dict with job status, result, and error information
    """
    job_id = self.request.id
    try:
        all_chunks = []
        processed_files = 0

        for file_path in file_paths:
            # Check for task revocation using supported method
            if self.is_aborted():  # Correct method for checking revocation
                logger.info(f"Job {job_id} was cancelled during processing")
                return {
                    "status": JobStatus.CANCELLED,
                    "result": None,
                    "error": "Job was cancelled by user",
                }

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
            return {
                "status": JobStatus.CANCELLED,
                "result": None,
                "error": "Job was cancelled by user after processing",
            }

        # Update job status
        return {
            "job_id": job_id,   # Include original job ID
            "status": JobStatus.COMPLETED,
            "result": [chunk.dict() for chunk in all_chunks],
            "error": "",
        }

    except Exception as e:
        logger.error(f"Job {job_id} failed: {e}")
        return {
            "job_id": job_id,   # Include original job ID
            "status": JobStatus.FAILED,
            "result": None,
            "error": str(e),
            "traceback": str(e.__traceback__) if e.__traceback__ else None
        }


@celery_app.task(bind=True, name="trigger_callback")
def trigger_callback_task(self, job_id: str, callback_url: str) -> None:
    """
    Task to trigger callback to notify about job completion.
    This task receives the original job ID and callback URL.
    """
    from app.main import trigger_callback

    # Run the async function in a new event loop
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    try:
        logger.info(f"Triggering callback for job: {job_id}")
        # Use the original job ID for callback
        loop.run_until_complete(trigger_callback(job_id, callback_url))
    except Exception as e:
        logger.error(f"Callback task failed for job {job_id}: {e}")
    finally:
        loop.close()
