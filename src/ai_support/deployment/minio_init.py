"""Idempotently create the private MLflow artifact bucket in MinIO."""

import os
import time

import boto3
from botocore.exceptions import BotoCoreError, ClientError


def ensure_bucket() -> None:
    endpoint = os.environ["MLFLOW_S3_ENDPOINT_URL"]
    bucket = os.getenv("AI_SUPPORT_MLFLOW_BUCKET", "mlflow")
    client = boto3.client(
        "s3",
        endpoint_url=endpoint,
        aws_access_key_id=os.environ["AWS_ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ["AWS_SECRET_ACCESS_KEY"],
        region_name="us-east-1",
    )
    last_error: Exception | None = None
    for attempt in range(20):
        try:
            existing = {item["Name"] for item in client.list_buckets().get("Buckets", [])}
            if bucket not in existing:
                client.create_bucket(Bucket=bucket)
            return
        except (BotoCoreError, ClientError, OSError) as error:
            last_error = error
            time.sleep(min(1 + attempt, 5))
    raise RuntimeError("MinIO did not become ready for bucket initialization") from last_error


if __name__ == "__main__":
    ensure_bucket()
