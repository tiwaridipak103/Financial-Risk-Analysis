import os
import pandas as pd
import sys
import joblib
from pathlib import Path
import pathlib
import pickle

PACKAGE_ROOT = Path(os.path.abspath(os.path.dirname(__file__))).parent
sys.path.append(str(PACKAGE_ROOT))



import pandas as pd
import numpy as np 
from prediction_model.config import config  
from prediction_model.processing.data_handling import load_dataset,save_pipeline,Save_column
import prediction_model.processing.preprocessing as pp 
import prediction_model.pipeline as pipe 
from sklearn import metrics
from sklearn import linear_model
import scipy.stats as stat
import sys

import prediction_model

PACKAGE_ROOT = pathlib.Path(prediction_model.__file__).resolve().parent
DATAPATH = os.path.join(PACKAGE_ROOT,'dataset_LGD_EAD')


# PD Model Estimation
def perform_PD_training():
    train_data = load_dataset(config.TRAIN_FILE)
    train_data = train_data.loc[: ,config.FEATURES_SELECTION]
    train_data = train_data.drop(config.REF_CATEGORIES, axis = 1)

    # Save column names to a list
    columns = train_data.columns.tolist()
    Save_column(columns,config.PD_COLUMNS)


    train_y = load_dataset(config.TRAIN_TARGET)
    train_y = train_y[config.TARGET]

    pipe.classification_pipeline.fit(train_data,train_y)
    save_pipeline(pipe.classification_pipeline,config.MODEL_NAME)


#LGD model training 
def perform_LGD_training():
    #Stage 1 – Logistics Regression
    filepath = os.path.join(DATAPATH,'lgd_inputs_stage_1_train.csv')
    train_data = pd.read_csv(filepath)

    train_data = train_data.loc[: ,config.LGD_EAD_FEATURES_SELECTION]
    train_data = train_data.drop(config.LGD_EAD_REF_CATEGORIES, axis = 1)

    # Save column names to a list
    columns = train_data.columns.tolist()
    Save_column(columns,config.LGD_EAD_COLUMNS)

    filepath = os.path.join(DATAPATH,'lgd_targets_stage_1_train.csv')
    train_y = pd.read_csv(filepath)
    train_y = train_y[config.LGD_TARGET_STAGE_1]

    pipe.classification_pipeline.fit(train_data,train_y)
    save_pipeline(pipe.classification_pipeline,config.MODEL_NAME_STAGE_1)

    #Stage 2 – Linear Regression
    filepath = os.path.join(DATAPATH,'lgd_inputs_stage_2_train.csv')
    train_data = pd.read_csv(filepath)

    train_data = train_data.loc[: ,config.LGD_EAD_FEATURES_SELECTION]
    train_data = train_data.drop(config.LGD_EAD_REF_CATEGORIES, axis = 1)

    # Save column names to a list
    columns = train_data.columns.tolist()
    Save_column(columns,config.LGD_EAD_COLUMNS)

    filepath = os.path.join(DATAPATH,'lgd_targets_stage_2_train.csv')
    train_y = pd.read_csv(filepath)
    train_y = train_y[config.LGD_TARGET_STAGE_2]

    reg_lgd_st_2 = linear_model.LinearRegression()
    # We create an instance of an object from the 'LinearRegression' class.
    reg_lgd_st_2.fit(train_data, train_y)

    X = train_data
    y = train_y
    sse = np.sum((reg_lgd_st_2.predict(X) - y) ** 2, axis=0) / float(X.shape[0] - X.shape[1])
    se = np.array([np.sqrt(np.diagonal(sse * np.linalg.inv(np.dot(X.T, X))))])
    reg_lgd_st_2.t = reg_lgd_st_2.coef_ / se
    reg_lgd_st_2.p_values = np.squeeze(2 * (1 - stat.t.cdf(np.abs(reg_lgd_st_2.t), y.shape[0] - X.shape[1])))

    save_pipeline(reg_lgd_st_2,config.MODEL_NAME_STAGE_2)


# EAD Model Estimation
def perform_EAD_training():
    filepath = os.path.join(DATAPATH,'ead_inputs_train.csv')
    train_data = pd.read_csv(filepath)

    train_data = train_data.loc[: ,config.LGD_EAD_FEATURES_SELECTION]
    train_data = train_data.drop(config.LGD_EAD_REF_CATEGORIES, axis = 1)

    # Save column names to a list
    columns = train_data.columns.tolist()
    Save_column(columns,config.LGD_EAD_COLUMNS)

    filepath = os.path.join(DATAPATH,'ead_targets_train.csv')
    train_y = pd.read_csv(filepath)
    train_y = train_y[config.EAD_TARGET]

    reg_ead = linear_model.LinearRegression()
    # We create an instance of an object from the 'LinearRegression' class.
    reg_ead.fit(train_data, train_y)

    X = train_data
    y = train_y
    sse = np.sum((reg_ead.predict(X) - y) ** 2, axis=0) / float(X.shape[0] - X.shape[1])
    se = np.array([np.sqrt(np.diagonal(sse * np.linalg.inv(np.dot(X.T, X))))])
    reg_ead.t = reg_ead.coef_ / se
    reg_ead.p_values = np.squeeze(2 * (1 - stat.t.cdf(np.abs(reg_ead.t), y.shape[0] - X.shape[1])))

    save_pipeline(reg_ead,config.EAD_MODEL_NAME)


if __name__=='__main__':
    perform_PD_training()
    perform_LGD_training()
    perform_EAD_training()
    