from zenml import pipeline 
from zenml.client import Client
from pipelines.training_pipeline import train_pipeline
from steps.config import ModelNameConfig


config = ModelNameConfig(model_name="RandomForestClassifier")

if __name__ == "__main__":
    print(Client().active_stack.experiment_tracker.get_tracking_uri()) # get the experiment tracker url with this
    train_pipeline(data_path="data/loan_approval_dataset.pq", config=config)

