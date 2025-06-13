
from mlpipeline.config import Config, setup_logging
from mlpipeline.data.data_processor import DataProcessor
from mlpipeline.model.model import LLMWrapper

def main():

    # MANUAL RETRAIN

    # load BQ dataset
    data = DataProcessor(
        project_id = Config.GCP_PROJECT_ID,
        dataset_id = Config.BQ_DATASET_ID,
        table_id = Config.BQ_TABLE_ID,
        start_date = None,
    )
    data.create_splits()
    print(data.df.shape, data.ds.shape, data.train_df.shape, data.val_df.shape, data.test_df.shape)

    # load model from gcs
    model = LLMWrapper(
        model_name=Config.MODEL_NAME,
        adapter_name=Config.ADAPTER_NAME,
        local_directory=Config.LOCAL_DIRECTORY,
        project_id=Config.GCP_PROJECT_ID,
        bucket_name=Config.GCS_BUCKET_NAME,
    )

    train_metrics = model.train(
        data_train=data.train_df,
        data_val=data.val_df
    )
    eval_metrics = model.evaluate(
        data_test=data.test_df,
    )


if __name__ == '__main__':

    setup_logging()

    main()

