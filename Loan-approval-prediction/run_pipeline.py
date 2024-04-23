from zenml import pipeline 
from pipelines.training_pipeline import train_pipeline
from steps.config import ModelNameConfig

config = ModelNameConfig(model_name="CatBoostClassifier")

if __name__ == "__main__":
    train_pipeline(data_path="data/loan_approval_dataset.pq", config=config)
    