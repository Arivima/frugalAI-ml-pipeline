import logging
from google.cloud import storage
from mlpipeline.config import Config, setup_logging
from google.cloud.storage import transfer_manager
from google.cloud.exceptions import GoogleCloudError
from google.cloud import bigquery
from datetime import datetime, timezone


logger = logging.getLogger(__name__)

class GCP():

    @staticmethod
    def load_adapter_gcs():
        try:
            logger.info("load_model_gcs")

            project_id = Config.GCP_PROJECT_ID
            if not project_id:
                raise ValueError("GCP_PROJECT_ID is not configured in Config.")
            client = storage.Client(project=project_id)
            logger.info('connected to client : %s', client.project)

            bucket_name = Config.GCS_BUCKET_NAME
            if not bucket_name:
                raise ValueError("GCS_BUCKET_NAME is not configured in Config.")
            bucket = client.bucket(bucket_name)
            logger.info('connected to bucket : %s', bucket.name)

            prefix = Config.ADAPTER_NAME + '/'
            logger.info(f'Looking with prefix : {prefix}')
            blobs = list(bucket.list_blobs(
                prefix=prefix,
                delimiter="/"
                ))
            logger.info(f'Number of blobs found: {len(blobs)}')
            if not blobs:
                raise ValueError("No files found")

            blob_names = [blob.name for blob in blobs]

            destination_directory = Config.DESTINATION_DIRECTORY
            results = transfer_manager.download_many_to_path(
                bucket, blob_names, destination_directory=destination_directory
            )

            for name, result in zip(blob_names, results):
                if isinstance(result, Exception):
                    logger.error("Failed to download %s due to exception: %s", name, result)
                else:
                    logger.info("Downloaded %s.", destination_directory + '/' + name)


            logger.info("✅ Adapter downloaded successfully from GCS.")

            return destination_directory

        except Exception as e:
            logger.exception(f"❌ Error downloading adapter from GCS: {e}. Will try to load from cache.")

    @staticmethod    
    def load_data_bq(start_date=None):
        try:
            project_id = Config.GCP_PROJECT_ID
            if not project_id:
                raise ValueError("GCP_PROJECT_ID is not configured in Config.")

            client = bigquery.Client(project=project_id)

            dataset_id = Config.BQ_DATASET_ID
            if not dataset_id:
                raise ValueError("BQ_DATASET_ID is not configured in Config.")

            table_id = Config.BQ_TABLE_ID
            if not table_id:
                raise ValueError("BQ_TABLE_ID is not configured in Config.")

            logger.info(f'BigQuery table : {project_id}.{dataset_id}.{table_id}')
            
            query = f"""
            SELECT 
                user_claim as text,
                predicted_category as label_pred,
                correct_category as label_true,
                created_at
            FROM `{dataset_id}.{table_id}`
            """
            
            if start_date:
                query += f" AND timestamp > '{start_date}'"
            
            df = client.query(query).to_dataframe()
            df['text'] = df['text'].str.strip()

            return df

        except GoogleCloudError as e:
            logger.error(f"Google Cloud error while inserting feedback: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error while inserting feedback: {e}")
            raise



if __name__ == '__main__':

    setup_logging()

    # load_model_gcs()

