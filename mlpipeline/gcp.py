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
    def load_adapter_gcs(
        project_id : str,
        bucket_name : str,
        adapter_name : str,
        local_directory : str,
    ):
        try:
            logger.info("load_model_gcs")

            logger.info(f"project_id : {project_id}")
            logger.info(f"bucket_name : {bucket_name}")
            logger.info(f"adapter_name : {adapter_name}")
            logger.info(f"local_directory : {local_directory}")

            if not project_id:
                raise ValueError("Missing project_id.")
            if not bucket_name:
                raise ValueError("Missing bucket_name.")
            if not adapter_name:
                raise ValueError("Missing adapter_name.")
            if not local_directory:
                raise ValueError("Missing local_directory.")

            client = storage.Client(project=project_id)
            logger.info('connected to client : %s', client.project)

            bucket = client.bucket(bucket_name)
            logger.info('connected to bucket : %s', bucket.name)

            prefix = adapter_name + '/'
            logger.info(f'Looking with prefix : {prefix}')
            blobs = list(bucket.list_blobs(
                prefix=prefix,
                delimiter="/"
                ))
            logger.info(f'Number of blobs found: {len(blobs)}')
            if not blobs:
                raise ValueError("No files found")

            blob_names = [blob.name for blob in blobs]

            results = transfer_manager.download_many_to_path(
                bucket, blob_names, local_directory=local_directory
            )

            for name, result in zip(blob_names, results):
                if isinstance(result, Exception):
                    logger.error("Failed to download %s due to exception: %s", name, result)
                else:
                    logger.info("Downloaded %s.", local_directory + '/' + name)


            logger.info("✅ Adapter downloaded successfully from GCS.")

            return local_directory

        except Exception as e:
            logger.exception(f"❌ Error downloading adapter from GCS: {e}. Will try to load from cache.")

    @staticmethod    
    def load_data_bq(
        project_id : str,
        dataset_id : str,
        table_id : str,
        start_date=None,
        ):
        try:
            logger.info(f"project_id : {project_id}")
            logger.info(f"dataset_id : {dataset_id}")
            logger.info(f"table_id : {table_id}")
            logger.info(f"start_date : {start_date}")
            
            if not project_id:
                raise ValueError("Missing project_id.")
            if not dataset_id:
                raise ValueError("Missing dataset_id.")
            if not table_id:
                raise ValueError("Missing table_id.")

            client = bigquery.Client(project=project_id)

            logger.info(f'BigQuery table : {project_id}.{dataset_id}.{table_id}')
            
            query = f"""
            SELECT 
                user_claim as text,
                predicted_category as label_pred,
                correct_category as label_true,
                assistant_explanation as explanation,
                created_at
            FROM `{dataset_id}.{table_id}`
            """
            
            if start_date:
                query += f" AND timestamp > '{start_date}'"
            
            df = client.query(query).to_dataframe()
            df['text'] = df['text'].str.strip()

            return df

        except GoogleCloudError as e:
            logger.error(f"Google Cloud error load_data_bq: {e}")
            raise
        except Exception as e:
            logger.error(f"Error load_data_bq: {e}")
            raise



if __name__ == '__main__':

    setup_logging()

    # load_model_gcs()

