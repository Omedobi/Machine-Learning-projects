import logging
from abc import ABC, abstractmethod
from catboost import CatBoostClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.base import ClassifierMixin

class Model(ABC):
    
    @abstractmethod
    def train(self, X_train, y_train, **kwargs) -> ClassifierMixin:
        pass
    
    
class XGBBoostModel(Model):

    def train(self, X_train, y_train, **kwargs) -> ClassifierMixin:
        try:
            xgb = XGBClassifier(**kwargs)
            xgb.fit(X_train,y_train)
            logging.info("Model training completed")
            return xgb
        except Exception as e:
            logging.error(f"Error in training XGboost model: {e}")
            raise e
         
        
class RandomForestModel(Model):
    
    def train(self, X_train, y_train, **kwargs) -> ClassifierMixin:
        try:
            rf = RandomForestClassifier(**kwargs)
            rf.fit(X_train, y_train)
            logging.info("Model training completed")
            return rf
        except Exception as e:
            logging.info(f"Error in training Random Forest model: {e}")
            raise e
            
            
    