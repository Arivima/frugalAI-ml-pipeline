import os
import logging

class Config:
    GCP_PROJECT_ID = os.getenv("GCP_PROJECT_ID", "")
    BQ_DATASET_ID = os.getenv("BQ_DATASET_ID", "")
    BQ_TABLE_ID = os.getenv("BQ_TABLE_ID", "")
    GCS_BUCKET_NAME = os.getenv("GCS_BUCKET_NAME", "")
    DESTINATION_DIRECTORY = os.getenv("DESTINATION_DIRECTORY", "")
    ADAPTER_NAME = os.getenv("ADAPTER_NAME", "")
    MODEL_NAME = os.getenv("MODEL_NAME", "")

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

