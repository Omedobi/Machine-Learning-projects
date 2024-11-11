import numpy as np
import pandas as pd
import logging
import json
from zenml import pipeline, step
from zenml.config import DockerSettings
from zenml.constants import DEFAULT_SERVICE_START_STOP_TIMEOUT
from zenml.integrations.constants import MLFLOW
from zenml.integrations.mlflow.model_deployers.mlflow_model_deployer import MLFlowModelDeployer
from zenml.integrations.mlflow.services import MLFlowDeploymentService
from zenml.integrations.mlflow.steps import mlflow_model_deployer_step
from zenml.steps import BaseParameters, Output

from steps.clean_data import clean_df
from steps.evaluation import evaluate_model
from steps.ingest_data import ingest_df
from steps.model_train import train_model
from pipelines.utlis import get_data_for_test

docker_settings = DockerSettings(required_integrations=[MLFLOW])

class DeploymentTriggerConfiguration(BaseParameters):
    """Deploment trigger Config"""
    min_accuracy: float = 0.7

@step(enable_cache=False)
def dynamic_importer() -> str:
   data = get_data_for_test()
   return data 
   
@step 
def deployment_trigger(
    accuracy: float,
    config: DeploymentTriggerConfiguration,
):
    """This implements a model deployment that takes account of the input model accuracy and 
    decides whether to deploy or not to deploy"""
    return accuracy > config.min_accuracy

class MLFlowDeploymentLoaderStepParameters(BaseParameters):
    pipeline_name: str
    step_name: str
    running: bool = True

@step
def predictor(
    service: MLFlowDeploymentService,
    data: str,
) -> np.ndarray:
    
    service.start(timeout=10)
    data = json.loads(data)
    data.pop('columns')
    data.pop('index')
    columns_for_df = [
        "no_of_dependents",
        "education", 
        "self_employed", 
        "income_annum",
        "loan_amount",
        "loan_term",
        "credit_score",
        "Movable_assets",
        "Immovable_assets", 
        # "luxury_assets_value", 
        # "bank_asset_value", 
    ]
    df = pd.DataFrame(data['data'], columns=columns_for_df)
    json_list = json.loads(json.dumps(list(df.T.to_dict().values())))
    data = np.array(json_list)
    prediction = service.predict(data)
    return prediction

@step(enable_cache=False)
def prediction_service_loader(
    pipeline_name: str,
    pipeline_step_name: str,
    running: bool = True,
    model_name: str = "model",
) -> MLFlowDeploymentService:
    # Get MLflow deployer stack component
    mlflow_model_deployer_component = MLFlowModelDeployer.get_active_model_deployer()

    # Fetch existing services with the same pipeline name, step name and model name
    existing_services = mlflow_model_deployer_component.find_model_server(
        pipeline_name=pipeline_name,
        pipeline_step_name=pipeline_step_name,
        model_name=model_name,
        running=running,
    )

    if not existing_services:
        logging.error(f"No MLflow deployment service found for pipeline {pipeline_name}, "
                      f"step {pipeline_step_name} and model {model_name}.")
        raise RuntimeError(
            f"No MLflow deployment service found for pipeline {pipeline_name}, "
            f"step {pipeline_step_name} and model {model_name}. "
            "Pipeline for the {model_name} model is currently not running."
        )
    
    logging.info(f"Found existing service: {existing_services[0].uuid}")
    return existing_services[0]

@pipeline(enable_cache=False, settings={"docker": docker_settings}) 

def continuous_deployment_pipeline(
    data_path: str,
    min_accuracy: float = 0.92,
    workers : int = 1,
    timeout: int = DEFAULT_SERVICE_START_STOP_TIMEOUT,
):
    df = ingest_df(data_path=data_path)
    X_train,X_test,y_train,y_test = clean_df(df)
    model = train_model(X_train, X_test, y_train, y_test)
    r2, mae, mse = evaluate_model(model, X_test, y_test)
    deployment_decision = deployment_trigger(r2)
    
    if deployment_decision:
        mlflow_model_deployer_step(
            model = model,
            deploy_decision = True,
            workers = workers,
            timeout = timeout
        )

@pipeline(enable_cache=False, settings={"docker":docker_settings})

def inference_pipeline(pipeline_name: str, pipeline_step_name: str):
    data = dynamic_importer()
    service = prediction_service_loader(
        pipeline_name=pipeline_name,
        pipeline_step_name=pipeline_step_name,
        running = False,
    )
    prediction = predictor(service=service, data=data)
    return prediction 
    
