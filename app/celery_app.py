from celery import Celery
from app.config import settings

# Initialize Celery app
celery_app = Celery(
    "deepdoc",
    broker=settings.REDIS_URL,
    backend=settings.REDIS_URL,
    include=["app.tasks"],
)

# Optional configuration
celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    task_track_started=True,
    task_time_limit=3600,  # 1 hour time limit for tasks
    worker_concurrency=2,  # Adjust based on your needs
    worker_max_tasks_per_child=100,  # Restart workers after this many tasks
)

if __name__ == "__main__":
    celery_app.start()
