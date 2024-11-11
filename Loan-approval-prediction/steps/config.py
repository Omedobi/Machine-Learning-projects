from zenml.steps import BaseParameters
from typing import List

class ModelNameConfig(BaseParameters):
    
    model_name: str = "RandomForestClassifier" #"XGBClassifier"
    
   