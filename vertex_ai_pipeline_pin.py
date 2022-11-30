######################################################################################################################################
#Title: vertex_ai_pipeline_pin.py
#Description: 
#We have taken a basic churn problem; the code for which can be found below:
#https://github.com/manan-bedi2908/Customer_Churn-Deployment/blob/master/Customer_Churn.ipynb 
#We have converted this code into a Vertex AI pipeline code. 
#Date: October 10, 2022         
######################################################################################################################################
from typing import NamedTuple
from datetime import datetime 
from kfp.v2 import compiler, dsl
from kfp.v2.dsl import pipeline, component, Artifact, Dataset, Input, Metrics, Model, Output, InputPath, OutputPath,Condition
from google.cloud import aiplatform
from google.cloud import aiplatform_v1
import pandas as pd
import os
######################################################################################################################################
PROJECT_ID = "<YOUR_PROJECT_ID>"
BUCKET_NAME = "<YOUR_BUCKET_NAME>" 
REGION = "<REGION_NAME>" 
PIPELINE_ROOT = f"{BUCKET_NAME}/pipeline_root/"
print(PIPELINE_ROOT)
######################################################################################################################################
@component(
   packages_to_install = ["google-cloud-bigquery", "pandas"],
   base_image = "python:3.9",
   output_component_file = "create_dataset.yaml"
)
def get_dataframe(
   output_data_path: OutputPath("Dataset")
):
   from google.cloud import storage
   import pandas as pd
   import numpy as np
   # Initialise a client
   storage_client = storage.Client()
   # Create a bucket object for our bucket
   bucket = storage_client.get_bucket('<YOUR_BUCKET_NAME>')
   # Create a blob object from the filepath
   blob = bucket.blob('Churn_Modelling.csv')
   # Download the file to a destination
   blob.download_to_filename('Churn_Modelling.csv')
   df = pd.read_csv('Churn_Modelling.csv')
   df.to_csv(output_data_path)
###################################################################################################################################### 
@component(
   packages_to_install = ["sklearn", "pandas", "joblib","google-cloud-bigquery"],
   base_image = "python:3.9",
   output_component_file = "model_component.yaml",
)
def sklearn_train(
   dataset: Input[Dataset],
   metrics: Output[Metrics],
   model: Output[Model]
) -> NamedTuple("output", [("deploy", str)]):
   import pandas as pd
   import numpy as np
   final_dataset = pd.read_csv(dataset.path)
   final_dataset = final_dataset[['CreditScore', 'Geography', 'Gender', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'HasCrCard', 'IsActiveMember', 'EstimatedSalary', 'Exited']]
   #print(final_dataset.head())
   # Converting the categorical variables into numerical and avoiding Dummy Varibale Trap
   final_df = pd.get_dummies(final_dataset, drop_first=True)
   # Splitting the Dataset into Dependent and Independent Variables
   X = final_df.iloc[:, [0,1,2,3,4,5,6,7,9,10,11]]
   y = final_df.iloc[:, 8].values
   # Splitting the dataset into Training and Testing Data
   from sklearn.model_selection import train_test_split
   X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2, random_state = 42)
   # Standardizing the Dataset
   from sklearn.preprocessing import StandardScaler
   sc = StandardScaler()
   X_train = sc.fit_transform(X_train)
   X_test = sc.transform(X_test)
   #print(X_train)
   print('Reached before feature importance')
   #Feature Importance
   from sklearn.ensemble import ExtraTreesRegressor
   skmodel = ExtraTreesRegressor()
   skmodel.fit(X,y)
   print(skmodel.feature_importances_)
   from sklearn.ensemble import RandomForestClassifier
   rf = RandomForestClassifier()
   rf.fit(X_train,y_train)
   y_pred = rf.predict(X_test)
   print('Reached after feature importance')
   from sklearn.metrics import accuracy_score, confusion_matrix
   #cm = confusion_matrix(y_test,y_pred)
   #print(cm)
   score = accuracy_score(y_test,y_pred)
   print('Accuracy Score: ')
   print(accuracy_score)
   # pickling the Model
   import json
   from datetime import datetime
   from google.cloud import storage
   storage_client = storage.Client()
   bucket = storage_client.get_bucket('<YOUR_BUCKET_NAME>')
   blob = bucket.blob('<YOUR_FILE_NAME>.json')
   blob.download_to_filename('<YOUR_FILE_NAME>.json')
   TIMESTAMP = datetime.now().strftime("%Y%m%d%H%M%S")
   print(TIMESTAMP)
   a_dict = {TIMESTAMP: score}
   with open('<YOUR_FILE_NAME>.json') as f:
       data = json.load(f)
   data.update(a_dict)
   with open('<YOUR_FILE_NAME>.json', 'w') as f:
       json.dump(data, f)
   blob.upload_from_filename('<YOUR_FILE_NAME>.json')
   from joblib import dump
   dump(skmodel, model.path + ".joblib")
   file = open('Customer_Churn_Prediction.pkl', 'wb')
   metrics.log_metric("accuracy",(score * 100.0))
   metrics.log_metric("framework", "Scikit Learn")
   metrics.log_metric("dataset_size", len(final_dataset))
   latest_model_threshold = data[max(data, key=int)]
   latest_model_threshold_str = str(latest_model_threshold < 0.75).lower()
   deploy = latest_model_threshold_str
   return (deploy,)
######################################################################################################################################
@component(
   packages_to_install = ["google-cloud-aiplatform"],
   base_image = "python:3.9",
   output_component_file = "deploy_component.yaml",
)
def deploy_model(
   model: Input[Model],
   project: str,
   region: str,
   vertex_endpoint: Output[Artifact],
   vertex_model: Output[Model]
):
   from google.cloud import aiplatform
 
   aiplatform.init(project = project, location = region)
 
   deployed_model = aiplatform.Model.upload(
       display_name = "<YOUR_PIPELINE_NAME>",
       artifact_uri = model.uri.replace("model", ""),
       serving_container_image_uri = "us-docker.pkg.dev/vertex-ai/prediction/sklearn-cpu.0-24:latest"
   )
   endpoint = deployed_model.deploy(machine_type = "n1-standard-4") 
   #Save data to the output params
   vertex_endpoint.uri = endpoint.resource_name
   vertex_model.uri = deployed_model.resource_name
   print('Done!')
###################################################################################################################################### 
@pipeline(
   # Default pipeline root. You can override it when submitting the pipeline.
   pipeline_root = PIPELINE_ROOT,
   # A name for the pipeline.
   name = "<YOUR_PIPELINE_NAME>",
)
def pipeline(
   output_data_path: str = "data.csv",
   project: str = PROJECT_ID,
   region: str = REGION
):
   dataset_task = get_dataframe()
 
   model_task = sklearn_train(
       dataset_task.output
   )
   with dsl.Condition(model_task.outputs["deploy"] == "true", name = "if_pipline_deploy"):
       deploy_task = deploy_model(
           model = model_task.outputs["model"],
           project = project,
           region = region
       )
######################################################################################################################################
compiler.Compiler().compile(
   pipeline_func = pipeline, package_path = "<YOUR_FILE_NAME>.json"
)
######################################################################################################################################  
TIMESTAMP = datetime.now().strftime("%Y%m%d%H%M%S")
run1 = aiplatform.PipelineJob(
   display_name = "<YOUR_PIPELINE_NAME>",
   template_path = "<YOUR_FILE_NAME>.json",
   job_id = "<YOUR_PIPELINE_NAME>-{0}".format(TIMESTAMP),
   enable_caching = True,
)
###################################################################################################################################### 
run1.submit()
######################################################################################################################################
