import logging
import pandas as pd
import numpy as np
import pyarrow

from sklearn.preprocessing import LabelEncoder
from abc import ABC, abstractmethod
from sklearn.model_selection import train_test_split
from typing import Union

class DataStrategy(ABC):
    
    @abstractmethod
    
    def handle_data(self, data: pd.DataFrame) -> Union[pd.DataFrame, pd.Series]:
        pass
        

class DataPreprocessStrategy(DataStrategy):
    
    def handle_data(self, data:pd.DataFrame) -> pd.DataFrame:
        
        try:
            
            data.columns = data.columns.str.strip()
            label_encoder = LabelEncoder()
            cat_cols = data.select_dtypes(include=['object']).columns
            for col in cat_cols:
                data[col] = label_encoder.fit_transform(data[col])
            
            return data
        except Exception as e:
            logging.error(f"Error in preprocessing data: {e}")
            raise e
        
class DataDivideStrategy(DataStrategy):
    
    def handle_data(self, data: pd.DataFrame) -> Union[pd.DataFrame, pd.Series]:
        
        try:
            X = data.drop(["loan_status"], axis=1)
            y = data["loan_status"]
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            return X_train, X_test, y_train, y_test
        except Exception as e:
            logging.error(f"Error in splitting the data: {e}")
            raise e
        
        
class DataCleaning:
    def __init__(self, data:pd.DataFrame, strategy:DataStrategy):
        self.data = data
        self.strategy = strategy
        
    def handle_data(self) -> Union[pd.DataFrame, pd.Series]:
        
        try:
            return self.strategy.handle_data(self.data)
        except Exception as e:
            logging.error(f"Error in handling the data {e}")
            raise e
        
if __name__ == "__main__":
    data = pd.read_parquet("data/loan_approval_dataset.pq")
    data_cleaning = DataCleaning(data, DataPreprocessStrategy())
    data_cleaning.handle_data()