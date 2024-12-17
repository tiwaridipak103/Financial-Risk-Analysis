from zenml.pipelines import pipeline
from steps.ingest_data import ingest_df
from steps.clean_data import clean_data
from steps.model_train import train_model
from steps.evaluation import evaluation
from steps.config import ModelNameConfig

import json

# from .utils import get_data_for_test
# import os

# import numpy as np
# import pandas as pd
#from materializer.custom_materializer import cs_materializer
# from steps.clean_data import clean_data
# from steps.evaluation import evaluation
# from steps.ingest_data import ingest_df
# from steps.model_train import train_model
from zenml import pipeline, step
# from zenml.config import DockerSettings
from zenml.constants import DEFAULT_SERVICE_START_STOP_TIMEOUT
# from zenml.integrations.constants import MLFLOW, TENSORFLOW
# from zenml.integrations.mlflow.model_deployers.mlflow_model_deployer import (
#     MLFlowModelDeployer,
# )
# from zenml.integrations.mlflow.services import MLFlowDeploymentService
# from zenml.integrations.mlflow.steps import mlflow_model_deployer_step
# from pydantic import BaseModel
# from steps.config import ModelNameConfig

# from .utils import get_data_for_test

# #docker_settings = DockerSettings(required_integrations=[MLFLOW])
# import pandas as pd


# @pipeline(enable_cache=False)
# def continuous_deployment_pipeline(data_path: str):
#     df = ingest_df(data_path)
#     x_train, x_test, y_train, y_test = clean_data(df)
#     model = train_model(x_train, x_test, y_train, y_test,config=ModelNameConfig()) 
#     mse, rmse = evaluation(model, x_test, y_test)

from zenml.integrations.mlflow.steps import mlflow_model_deployer_step
from pydantic import BaseModel
class DeploymentTriggerConfig(BaseModel):
    """Parameters that are used to trigger the deployment"""

    min_accuracy: float = 0.9

@step
def deployment_trigger(
    accuracy: float,
    config: DeploymentTriggerConfig,
) -> bool:
    """Implements a simple model deployment trigger that looks at the
    input model accuracy and decides if it is good enough to deploy"""

    return accuracy > config.min_accuracy

@pipeline(enable_cache=False)
def continuous_deployment_pipeline(data_path : str,
    min_accuracy: float = 0.9,
    workers: int = 1,
    timeout: int = DEFAULT_SERVICE_START_STOP_TIMEOUT):
    df = ingest_df(data_path)
    x_train, x_test, y_train, y_test = clean_data(df)
    model = train_model(x_train, x_test, y_train, y_test,config=ModelNameConfig()) 
    mse, rmse = evaluation(model, x_test, y_test)
    deployment_decision = deployment_trigger(accuracy=mse,config = DeploymentTriggerConfig())
    mlflow_model_deployer_step(
        model=model,
        deploy_decision=deployment_decision,
        workers=workers,
        timeout=timeout,
    )



# import json

# # from .utils import get_data_for_test
# import os

# import numpy as np
# import pandas as pd
# from materializer.custom_materializer import cs_materializer
# from steps.clean_data import clean_data
# from steps.evaluation import evaluation
# from steps.ingest_data import ingest_df
# from steps.model_train import train_model
# from zenml import pipeline, step
# from zenml.config import DockerSettings
# from zenml.constants import DEFAULT_SERVICE_START_STOP_TIMEOUT
# from zenml.integrations.constants import MLFLOW, TENSORFLOW
# from zenml.integrations.mlflow.model_deployers.mlflow_model_deployer import (
#     MLFlowModelDeployer,
# )
# from zenml.integrations.mlflow.services import MLFlowDeploymentService
# from zenml.integrations.mlflow.steps import mlflow_model_deployer_step
# from pydantic import BaseModel
# from steps.config import ModelNameConfig

# from .utils import get_data_for_test

# #docker_settings = DockerSettings(required_integrations=[MLFLOW])
# import pandas as pd

# requirements_file = os.path.join(os.path.dirname(__file__), "requirements.txt")



# class DeploymentTriggerConfig(BaseModel):
#     """Parameters that are used to trigger the deployment"""

#     min_accuracy: float = 0.9

# from pydantic import BaseModel, Field, validator

# class MyModel(BaseModel):
#     data: dict  # Convert DataFrame to dictionary

#     @validator("data", pre=True, always=True)
#     def validate_and_convert_dataframe(cls, value):
#         if isinstance(value, pd.DataFrame):
#             return value.to_dict()
#         elif isinstance(value, dict):
#             return value  # Allow pre-converted data
#         else:
#             raise ValueError("The 'data' field must be a pandas DataFrame or dict.")




# @step
# def deployment_trigger(
#     accuracy: float,
#     config: DeploymentTriggerConfig,
# ) -> bool:
#     """Implements a simple model deployment trigger that looks at the
#     input model accuracy and decides if it is good enough to deploy"""

#     return accuracy > config.min_accuracy

# #@pipeline(enable_cache=False, settings={"docker": docker_settings})
# @pipeline(enable_cache=False)
# def continuous_deployment_pipeline(
#     data_path : str,
#     min_accuracy: float = 0.9,
#     workers: int = 1,
#     timeout: int = DEFAULT_SERVICE_START_STOP_TIMEOUT,
# ):
#     # Link all the steps artifacts together
#     data_path = r"C:\Users\DELL\Desktop\Mlops\data\olist_customers_dataset.csv"
#     df = ingest_df(data_path)
#     #validate_df = MyModel(data=df)
#     df = pd.read_csv(data_path)
#     x_train, x_test, y_train, y_test = clean_data(df)
#     model = train_model(x_train, x_test, y_train, y_test,config=ModelNameConfig())
#     mse, rmse = evaluation(model, x_test, y_test)
#     deployment_decision = deployment_trigger(accuracy=mse,config = DeploymentTriggerConfig())
#     mlflow_model_deployer_step(
#         model=model,
#         deploy_decision=deployment_decision,
#         workers=workers,
#         timeout=timeout,
#     )

