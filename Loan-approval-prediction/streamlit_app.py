import json
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

from pipelines.deployment_pipeline import prediction_service_loader
from run_deployment import run_deployment


def app():
    st.title("End-to-End Customer Loan Approval Prediction Pipeline with ZenML")
    
    # Display high-level and whole pipeline images
    continuous_pipeline_image = Image.open("_assets/continuous_pipepline_update.png")
    st.image(continuous_pipeline_image, caption="Continuous Pipeline")
    
    whole_pipeline_image = Image.open("_assets/training_and_deployment_pipeline_update.png")
    
    st.markdown(
    """
    #### Problem statement
    the objective here is to predict the customer loan approval status
    """
    )
    
    st.image(whole_pipeline_image, caption="Complete Pipeline Overview")
    st.markdown(
    """
     The diagram above represents the entire pipeline: data ingestion, cleaning, model training, 
     evaluation, and prediction using inference data.
       
    """
    )
    
    st.markdown(
    """#### Feature Descriptions
    This app predicts the loan approval status based on the following features:
     | Variable               | Description                                             |
     | ---------------------  | ------------------------------------------------------- |
     |loan_id                 | Unique loan ID                                          |
     |no_of_dependents        | Number of dependents of the applicant                   |
     |education               | Education level of the applicant                        |
     |self_employed           | If the applicant is self-employed or not                |
     |income_annum            | Annual income of the applicant                          |
     |loan_amount             | Loan amount requested by the applicant                  |
     |loan_tenure             | Tenure of the loan requested by the applicant (in Years)|
     |credit_score            | credit score of the applicant                           |
     |residential_asset_value | Value of the residential asset of the applicant         |
     |commercial_asset_value  | Value of the commercial asset of the applicant          |
     |luxury_asset_value      | Value of the luxury asset of the applicant              |
     |bank_assets_value       | Value of the bank asset of the applicant                |
     |loan_status             | Status of the loan (Approved/Rejected)                  |
    """
    )
    # sidebar inputs for features
    no_of_dependents = st.sidebar.slider("No of Dependents", min_value=0, max_value=20, step=1)
    education = st.sidebar.selectbox("Education", ("Graduate", "Not Graduate"))
    self_employed = st.sidebar.selectbox("Self Employed", ("Yes", "No"))
    income_annum = st.sidebar.slider("Annual Income", min_value=0, max_value=200000, step=1000)
    loan_amount = st.number_input("Loan amount", min_value=0)
    loan_tenure = st.sidebar.selectbox("Loan tenure", ("3 Months", "6 months", "9 months", "1 year"))
    credit_score = st.number_input("Credit score", min_value=0.0, max_value=1000, step=1)
    residential_asset_value = st.number_input("Resident asset value", min_value=0)
    commercial_asset_value = st.number_input("Commercial asset value", min_value=0)
    luxury_asset_value = st.number_input("Luxury asset value", min_value=0)
    bank_assets_value = st.number_input("Bank asset value", min_value=0)
    loan_id = st.number_input("loan ID", min_value=0)
    
    # Predict button logic
    if st.button("Predict"):
        service = prediction_service_loader(
            pipeline_name = "continuous_deployment_pipeline",
            pipeline_step_name = "mlflow_model_deployer_step",
            running = False,
        )
        
        if service is None:
            st.write("No service found. Running pipeline deployment to create a prediction service.")
            run_deployment()
        else:
            # Dataframe for input features   
            df = pd.DataFrame({
                "no_of_dependents":[no_of_dependents],
                "education":[education],
                "self_employed":[self_employed],
                "income_annum":[income_annum],
                "loan_amount":[loan_amount],
                "loan_tenure":[loan_tenure],
                "credit_score":[credit_score],
                "residential_asset_value":[residential_asset_value],
                "commercial_asset_value":[commercial_asset_value],
                "luxury_asset_value":[luxury_asset_value],
                "bank_assets_value":[bank_assets_value],
                "loan_id":[loan_id],
                # "loan_status":[loan_status],
                })
            
            # convert the dataframe to JSON for prediction
            data_json = df.to_json(orient="records")
            data = np.array(json.loads(data_json))
            
            # Predict and display the result
            pred = service.predict(data)
            st.success(f"Your loan approval status based on the provided details: {pred}")
       
        # Button for displaying model performance
        if st.button("Show model results"):
            st.write("Model performance comparison between Random Forest and XGBoost:")
            
            # Model performance dataframe
            result_df = pd.DataFrame({
                    "Models": ["Random Forest", "XGBoost Classifier"],
                    "MSE":[1.804, 1.781],
                    "RMSE":[1.343, 1.335],
                })
            
            st.dataframe(result_df)
            
            st.write("Inference data to check model performance:")
            inference_image = Image.open("_assets/inference_pipeline_update.png")
            st.image(inference_image, caption="Inference pipeline")
            
            
if __name__ == "__main__":
    app()