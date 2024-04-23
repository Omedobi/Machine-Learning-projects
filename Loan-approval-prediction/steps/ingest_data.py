import logging
import pandas as pd
from zenml import step

class IngestData:
    """
    ingesting the data into the data_path
    """
    def __init__(self, data_path:str):
        self.data_path = data_path
        
    def get_data(self):
        logging.info("Ingesting data from %s", self.data_path)
        return pd.read_parquet(self.data_path)
    
@step
def ingest_df(data_path: str) -> pd.DataFrame:
    try:
        ingest_obj = IngestData(data_path)
        df = ingest_obj.get_data()
        return df
    except Exception as e:
        logging.error("Error while ingesting data: %s", e)
        raise e