data_dev : 
	UV_ENV_FILE=".env" uv run python -m mlpipeline.data.data_processor

model_dev :
	UV_ENV_FILE=".env" uv run python -m mlpipeline.model.model