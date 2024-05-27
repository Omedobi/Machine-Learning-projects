import logging
import numpy as np
from abc import ABC, abstractmethod
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class Evaluation(ABC):
    
    @abstractmethod
    def calculate_scores(self, y_true: np.ndarray, y_pred: np.ndarray) :
        """Calculate the scores for the model
        Args:
            y_true (np.ndarray): True labels
            y_pred (np.ndarray): Predicted labels
        returns:
            None
        """
        pass
    
class MSE(Evaluation):
    def calculate_scores(self, y_true: np.ndarray, y_pred: np.ndarray) :
        try:
            logging.info("Calculating MSE")
            mse = mean_squared_error(y_true, y_pred)
            logging.info("MSE: {:.3f}".format(mse))
            return mse
        except Exception as e:
            logging.error(f"Error in calculating MSE:{e}")
            raise e
        
class R2(Evaluation):
    def calculate_scores(self, y_true: np.ndarray, y_pred: np.ndarray) :
        try:
            logging.info("Calculating R2 Score")
            r2 = r2_score(y_true, y_pred)
            logging.info("R2 Score: {:.3f}".format(r2))
            return r2
        except Exception as e:
            logging.error(f"Error in calculating R2 Score:{e}")
            raise e
        
class MAE(Evaluation):
    def calculate_scores(self, y_true: np.ndarray, y_pred: np.ndarray) :
        try:
            logging.info("Calculating MAE")
            mae = mean_absolute_error(y_true, y_pred)
            logging.info("MAE: {:.3f}".format(mae))
            return mae
        except Exception as e:
            logging.error(f"Error in calculating MAE:{e}")
            raise e