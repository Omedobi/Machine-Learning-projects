import logging
import pandas as pd
from zenml import step
from src.model_dev import XGBBoostModel
from src.model_dev import RandomForestModel
from .config import ModelNameConfig
from sklearn.base import ClassifierMixin

@step
def train_model(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    config: ModelNameConfig,) -> ClassifierMixin:
    
    try:
        model = None
        if config.model_name == "XGBClassifier":
            model = XGBBoostModel()
        elif config.model_name == "RandomForestClassifier":
            model = RandomForestModel()
        else:
            raise ValueError(f"Model {config.model_name} not supported ")
        
        if model:
            trained_model = model.train(X_train, y_train)
            return trained_model
    
    except Exception as e:
        logging.error(f"Error in training {config.model_name} model: {e}")
        raise e
    