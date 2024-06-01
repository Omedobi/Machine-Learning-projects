import logging
import pandas as pd
import mlflow
from zenml import step
from zenml.client import Client
from src.model_dev import XGBBoostModel
from src.model_dev import RandomForestModel
from .config import ModelNameConfig
from sklearn.base import ClassifierMixin


experiment_tracker = Client().active_stack.experiment_tracker

@step(experiment_tracker = experiment_tracker.name)
def train_model(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    config: ModelNameConfig,) -> ClassifierMixin:
    
    try:
        model = None
        if config.model_name == "XGBClassifier":
            mlflow.sklearn.autolog()
            model = XGBBoostModel()
        elif config.model_name == "RandomForestClassifier":
            mlflow.sklearn.autolog()
            model = RandomForestModel()
        else:
            raise ValueError(f"Model {config.model_name} not supported ")
        
        if model:
            trained_model = model.train(X_train, y_train)
            return trained_model
    
    except Exception as e:
        logging.error(f"Error in training {config.model_name} model: {e}")
        raise e
    