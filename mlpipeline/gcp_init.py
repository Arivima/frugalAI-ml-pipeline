from google.cloud import bigquery, storage
from google.cloud.exceptions import NotFound
from datetime import datetime
from mlpipeline.config import Config, setup_logging
import logging
import os

logger = logging.getLogger(__name__)


schema = [
    bigquery.SchemaField("created_at", "TIMESTAMP", mode="REQUIRED"),
    bigquery.SchemaField("user_claim", "STRING", mode="REQUIRED"),
    bigquery.SchemaField("predicted_category", "INTEGER", mode="REQUIRED"),
    bigquery.SchemaField("correct_category", "INTEGER", mode="REQUIRED"),
    bigquery.SchemaField("assistant_explanation", "STRING", mode="NULLABLE"),
]


def ensure_bq_resources():
    logger.info('\nChecking Big Query ressources')
    bq_client = bigquery.Client(project=Config.GCP_PROJECT_ID)

    # Check that dataset exists, else create it
    try:
        bq_client.get_dataset(Config.BQ_DATASET_ID)
        print(f"Dataset {Config.BQ_DATASET_ID} already exists.")
    except NotFound:
        dataset = bigquery.Dataset(f"{Config.GCP_PROJECT_ID}.{Config.BQ_DATASET_ID}")
        dataset.location = Config.GCP_REGION
        bq_client.create_dataset(dataset)
        print(f"Dataset {Config.BQ_DATASET_ID} created.")

    # Check that table exists, else create it
    table_ref = f"{Config.GCP_PROJECT_ID}.{Config.BQ_DATASET_ID}.{Config.BQ_TABLE_ID}"
    try:
        table = bq_client.get_table(table_ref)
        print(f"Table {Config.BQ_TABLE_ID} already exists.")
        print(f"Schema for table {table_ref}:")
        for field in table.schema:
            print(f" - {field.name}: {field.field_type} ({field.mode})")

    except NotFound:
        table = bigquery.Table(table_ref, schema=schema)
        bq_client.create_table(table)
        print(f"Table {Config.BQ_TABLE_ID} created.")
        print(f"Schema for table {table_ref}:")
        for field in table.schema:
            print(f" - {field.name}: {field.field_type} ({field.mode})")



def ensure_gcs_bucket():
    logger.info('\nChecking GCS ressources')
    storage_client = storage.Client(project=Config.GCP_PROJECT_ID)
    try:
        storage_client.get_bucket(Config.GCS_BUCKET_NAME)
        print(f"Bucket {Config.GCS_BUCKET_NAME} already exists.")
    except NotFound:
        bucket = storage_client.bucket(Config.GCS_BUCKET_NAME)
        storage_client.create_bucket(bucket, location=Config.GCP_REGION)
        print(f"Bucket {Config.GCS_BUCKET_NAME} created.")
        print(f"Please load production model to GCS.")



if __name__ == "__main__":
    setup_logging()

    ensure_bq_resources()
    ensure_gcs_bucket()
