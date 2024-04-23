import logging
from abc import ABC, abstractmethod
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

class Model(ABC):
    
    @abstractmethod
    
    def train(self, X_train, y_train, **kwargs):
        
        pass
    
    
class CatBoostModel(Model):

    def train(self, X_train, y_train, **kwargs):
        
        try:
            
            cat_b = CatBoostClassifier(**kwargs)
            cat_b.fit(X_train,y_train)
            logging.info("Model training completed")
            return cat_b
    
        except Exception as e:
            logging.error(f"Error in training model: {e}")
            raise e
        
        
class RandomForestModel(Model):
    
    def train(self, X_train, y_train, **kwargs):
        
        try:
            
            rf = RandomForestClassifier(**kwargs)
            rf.fit(X_train, y_train)
            logging.info("Model training completed")
            return rf
        except Exception as e:
            logging.info(f"Error in training model: {e}")
            raise e
            
            
    