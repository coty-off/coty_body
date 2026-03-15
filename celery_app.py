from __future__ import annotations

import os
from celery import Celery
from dotenv import load_dotenv

load_dotenv(".env")

REDIS_BROKER: str = os.getenv("REDIS_BROKER", "redis://redis:6379/0")
REDIS_BACKEND: str = os.getenv("REDIS_BACKEND", "redis://redis:6379/1")

celery = Celery(
    "coty_body_worker",
    broker=REDIS_BROKER,
    backend=REDIS_BACKEND,
)

celery.conf.update(
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    enable_utc=True,
    broker_connection_retry_on_startup=True,
    # Подтверждать задачу только после завершения.
    # Если воркер упадёт в середине — задача вернётся в очередь.
    task_acks_late=True,
    task_reject_on_worker_lost=True,
    # Хранить результат 1 час — клиент успеет забрать через polling
    result_expires=3600,
)