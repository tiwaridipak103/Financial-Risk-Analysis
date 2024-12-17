import warnings
warnings.filterwarnings('ignore')

import os
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import sys

PACKAGE_ROOT = Path(os.path.abspath(os.path.dirname(__file__))).parent
sys.path.append(str(PACKAGE_ROOT))

from prediction_model.config import config  
from prediction_model.processing.data_handling import load_pipeline,load_dataset

filepath = os.path.join(config.DATAPATH,'loan_data_2015_test.csv')

loan_data_2015_preprocessed = pd.read_csv(filepath)

#############################--------Probability of Default------------------##################

loan_data_2015_preprocessed['mths_since_last_delinq'].fillna(0, inplace = True)
loan_data_2015_preprocessed['mths_since_last_record'].fillna(0, inplace=True)

loan_data_2015_preprocessed = loan_data_2015_preprocessed.loc[: ,config.FEATURES_SELECTION]
loan_data_2015_preprocessed = loan_data_2015_preprocessed.drop(config.REF_CATEGORIES, axis = 1)

model_loaded = load_pipeline(config.MODEL_NAME)

# Access the Logistic Regression model from the pipeline
logistic_model = model_loaded.named_steps['LogisticClassifier']

# We apply the PD model to caclulate estimated default probabilities.
loan_data_2015_preprocessed['PD'] = logistic_model.model.predict_proba(loan_data_2015_preprocessed)[: ][: , 0]


###################################-----Loss Given Default----###############################################
loan_data_preprocessed_lgd_ead = pd.read_csv(filepath)

loan_data_preprocessed_lgd_ead['mths_since_last_delinq'].fillna(0, inplace = True)
loan_data_preprocessed_lgd_ead['mths_since_last_record'].fillna(0, inplace=True)

model_loaded = load_pipeline(config.MODEL_NAME_STAGE_1)
# Access the Logistic Regression model from the pipeline
logistic_model = model_loaded.named_steps['LogisticClassifier']

# Here we keep only the variables we need for the model.
loan_data_preprocessed_lgd_ead = loan_data_preprocessed_lgd_ead[config.LGD_EAD_FEATURES_SELECTION]

# Here we remove the dummy variable reference categories.
loan_data_preprocessed_lgd_ead = loan_data_preprocessed_lgd_ead.drop(config.LGD_EAD_REF_CATEGORIES, axis = 1)

# We apply the stage 1 LGD model and calculate predicted values.
loan_data_2015_preprocessed['recovery_rate_st_1'] = logistic_model.model.predict(loan_data_preprocessed_lgd_ead)

Linear_model_loaded = load_pipeline(config.MODEL_NAME_STAGE_2)

# We apply the stage 2 LGD model and calculate predicted values.
loan_data_2015_preprocessed['recovery_rate_st_2'] = Linear_model_loaded.predict(loan_data_preprocessed_lgd_ead)

# We combine the predicted values from the stage 1 predicted model and the stage 2 predicted model
# to calculate the final estimated recovery rate.
loan_data_2015_preprocessed['recovery_rate'] = loan_data_2015_preprocessed['recovery_rate_st_1'] * loan_data_2015_preprocessed['recovery_rate_st_2']

# We set estimated recovery rates that are greater than 1 to 1 and  estimated recovery rates that are less than 0 to 0.
loan_data_2015_preprocessed['recovery_rate'] = np.where(loan_data_2015_preprocessed['recovery_rate'] < 0, 0, loan_data_2015_preprocessed['recovery_rate'])
loan_data_2015_preprocessed['recovery_rate'] = np.where(loan_data_2015_preprocessed['recovery_rate'] > 1, 1, loan_data_2015_preprocessed['recovery_rate'])

# We calculate estimated LGD. Estimated LGD equals 1 - estimated recovery rate.
loan_data_2015_preprocessed['LGD'] = 1 - loan_data_2015_preprocessed['recovery_rate']

###################################-----Exposure at Default----###############################################

Linear_model_loaded = load_pipeline(config.EAD_MODEL_NAME)

# We apply the EAD model to calculate estimated credit conversion factor.
loan_data_2015_preprocessed['CCF'] = Linear_model_loaded.predict(loan_data_preprocessed_lgd_ead)

# We set estimated CCF that are greater than 1 to 1 and  estimated CCF that are less than 0 to 0.
loan_data_2015_preprocessed['CCF'] = np.where(loan_data_2015_preprocessed['CCF'] < 0, 0, loan_data_2015_preprocessed['CCF'])
loan_data_2015_preprocessed['CCF'] = np.where(loan_data_2015_preprocessed['CCF'] > 1, 1, loan_data_2015_preprocessed['CCF'])

# We calculate estimated EAD. Estimated EAD equals estimated CCF multiplied by funded amount.
loan_data_2015_preprocessed['EAD'] = loan_data_2015_preprocessed['CCF'] * loan_data_preprocessed_lgd_ead['funded_amnt']

# We calculate Expected Loss. EL = PD * LGD * EAD.
loan_data_2015_preprocessed['EL'] = loan_data_2015_preprocessed['PD'] * loan_data_2015_preprocessed['LGD'] * loan_data_2015_preprocessed['EAD']

# Total Expected Loss as a proportion of total funded amount for all loans.
print(f"Total Expected Loss as a proportion of total funded amount for all loans : {loan_data_2015_preprocessed['EL'].sum() / loan_data_preprocessed_lgd_ead['funded_amnt'].sum()}")

# classification_pipeline = load_pipeline(config.MODEL_NAME)

# def generate_predictions(data_input):
#     data = pd.DataFrame(data_input)
#     pred = classification_pipeline.predict(data[config.FEATURES])
#     output = np.where(pred==1,'Y','N')
#     result = {"prediction":output}
#     return result

# def generate_predictions():
#     test_data = load_dataset(config.TEST_FILE)
#     pred = classification_pipeline.predict(test_data[config.FEATURES])
#     output = np.where(pred==1,'Y','N')
#     print(output)
#     #result = {"Predictions":output}
#     return output

# if __name__=='__main__':
#     generate_predictions()