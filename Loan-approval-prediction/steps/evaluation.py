import logging
import pandas as pd
from zenml import step
from src.evaluation import MSE, MAE, R2
from typing_extensions import Annotated
from typing import Tuple
from sklearn.base import ClassifierMixin



@step
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
        
        r2_class = R2()
        r2 = r2_class.calculate_scores(y_test, prediction)
        
        mae_class = MAE()
        mae = mae_class.calculate_scores(y_test, prediction)
        
        return mse, r2, mae
   except Exception as e:
       logging.error(f"Error in evaluating model: {e}")
       raise e
