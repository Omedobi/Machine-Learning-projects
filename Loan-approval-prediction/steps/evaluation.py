import logging
import pandas as pd
import mlflow
from zenml import step
from zenml.client import Client
from src.evaluation import MSE, MAE, R2
from typing_extensions import Annotated
from typing import Tuple
from sklearn.base import ClassifierMixin

experiment_tracker = Client().active_stack.experiment_tracker

@step(experiment_tracker=experiment_tracker.name)
def evaluate_model(
    model: ClassifierMixin,               
    X_test: pd.DataFrame,
    y_test: pd.DataFrame,) -> Tuple[
        Annotated[float, "r2"],
        Annotated[float, "mae"],
        Annotated[float, "mse"]
    ]:
   try: 
        prediction = model.predict(X_test)
        mse_class = MSE()
        mse = mse_class.calculate_scores(y_test, prediction)
        mlflow.log_metric("mse", mse)
        r2_class = R2()
        r2 = r2_class.calculate_scores(y_test, prediction)
        mlflow.log_metric("r2", r2)
        mae_class = MAE()
        mae = mae_class.calculate_scores(y_test, prediction)
        mlflow.log_metric("mae", mae)
        return mse, r2, mae
   except Exception as e:
       logging.error(f"Error in evaluating model: {e}")
       raise e
