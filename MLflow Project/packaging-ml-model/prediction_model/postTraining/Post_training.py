import os
import pandas as pd
import sys
import joblib
from pathlib import Path
import pandas as pd
import numpy as np 
import pickle


PACKAGE_ROOT = Path(os.path.abspath(os.path.dirname(__file__))).parent.parent
sys.path.append(str(PACKAGE_ROOT))

from prediction_model.config import config  
from prediction_model.processing.data_handling import load_pipeline,Load_column

###################################-----Probability of Default----###############################################

model_loaded = load_pipeline(config.MODEL_NAME)

feature_name = Load_column(config.COLUMN_NAME)

# Access the Logistic Regression model from the pipeline
logistic_model = model_loaded.named_steps['LogisticClassifier']

p_values = logistic_model.p_values
p_values = np.append(np.nan, np.array(p_values))


summary_table = pd.DataFrame(columns = ['Feature name'], data = feature_name)
summary_table['Coefficients'] = np.transpose(logistic_model.coef_)
summary_table.index = summary_table.index + 1
summary_table.loc[0] = ['Intercept', logistic_model.intercept_[0]]
summary_table = summary_table.sort_index()
summary_table['p_values'] = p_values


df_ref_categories = pd.DataFrame(config.REF_CATEGORIES, columns = ['Feature name'])
df_ref_categories['Coefficients'] = 0
df_ref_categories['p_values'] = np.nan

df_scorecard = pd.concat([summary_table, df_ref_categories])
df_scorecard = df_scorecard.reset_index()

df_scorecard['Original feature name'] = df_scorecard['Feature name'].str.split(':').str[0]

min_score = 300
max_score = 850

min_sum_coef = df_scorecard.groupby('Original feature name')['Coefficients'].min().sum()
max_sum_coef = df_scorecard.groupby('Original feature name')['Coefficients'].max().sum()

df_scorecard['Score - Calculation'] = df_scorecard['Coefficients'] * (max_score - min_score) / (max_sum_coef - min_sum_coef)
df_scorecard['Score - Calculation'][0] = ((df_scorecard['Coefficients'][0] - min_sum_coef) / (max_sum_coef - min_sum_coef)) * (max_score - min_score) + min_score

df_scorecard['Score - Preliminary'] = df_scorecard['Score - Calculation'].round()

df_scorecard['Difference'] = df_scorecard['Score - Preliminary'] - df_scorecard['Score - Calculation']

df_scorecard['Score - Final'] = df_scorecard['Score - Preliminary']
df_scorecard['Score - Final'][77] = 16

save_path = os.path.join(config.SAVE_MODEL_PATH,config.SCORECARD_PD)

df_scorecard.to_csv(save_path)

###################################-----Loss Given Default----###############################################
#Stage 1 – Logistics Regression
model_loaded = load_pipeline(config.MODEL_NAME_STAGE_1)

feature_name = Load_column(config.LGD_EAD_COLUMNS)

# Access the Logistic Regression model from the pipeline
logistic_model = model_loaded.named_steps['LogisticClassifier']

summary_table = pd.DataFrame(columns = ['Feature name'], data = feature_name)
# Creates a dataframe with a column titled 'Feature name' and row values contained in the 'feature_name' variable.
summary_table['Coefficients'] = np.transpose(logistic_model.coef_)
# Creates a new column in the dataframe, called 'Coefficients',
# with row values the transposed coefficients from the 'LogisticRegression' object.
summary_table.index = summary_table.index + 1
# Increases the index of every row of the dataframe with 1.
summary_table.loc[0] = ['Intercept', logistic_model.intercept_[0]]
# Assigns values of the row with index 0 of the dataframe.
summary_table = summary_table.sort_index()
# Sorts the dataframe by index.
p_values = logistic_model.p_values
# We take the result of the newly added method 'p_values' and store it in a variable 'p_values'.
p_values = np.append(np.nan,np.array(p_values))
# We add the value 'NaN' in the beginning of the variable with p-values.
summary_table['p_values'] = p_values
# In the 'summary_table' dataframe, we add a new column, called 'p_values', containing the values from the 'p_values' variable.

save_path = os.path.join(config.SAVE_MODEL_PATH,config.SCORECARD_LGD_STAGE_1)

summary_table.to_csv(save_path)

#Stage 2 – Linear Regression
model_loaded = load_pipeline(config.MODEL_NAME_STAGE_2)

feature_name = Load_column(config.LGD_EAD_COLUMNS)

summary_table = pd.DataFrame(columns = ['Feature name'], data = feature_name)
# Creates a dataframe with a column titled 'Feature name' and row values contained in the 'feature_name' variable.
summary_table['Coefficients'] = np.transpose(model_loaded.coef_)
# Creates a new column in the dataframe, called 'Coefficients',
# with row values the transposed coefficients from the 'LogisticRegression' object.
summary_table.index = summary_table.index + 1
# Increases the index of every row of the dataframe with 1.
summary_table.loc[0] = ['Intercept', model_loaded.intercept_]
# Assigns values of the row with index 0 of the dataframe.
summary_table = summary_table.sort_index()
# Sorts the dataframe by index.
p_values = model_loaded.p_values
# We take the result of the newly added method 'p_values' and store it in a variable 'p_values'.
p_values = np.append(np.nan,np.array(p_values))
# We add the value 'NaN' in the beginning of the variable with p-values.
summary_table['p_values'] = p_values.round(3)
# In the 'summary_table' dataframe, we add a new column, called 'p_values', containing the values from the 'p_values' variable.
save_path = os.path.join(config.SAVE_MODEL_PATH,config.SCORECARD_LGD_STAGE_2)

summary_table.to_csv(save_path)


###################################-----Exposure at Default----###############################################
model_loaded = load_pipeline(config.EAD_MODEL_NAME)

feature_name = Load_column(config.LGD_EAD_COLUMNS)

summary_table = pd.DataFrame(columns = ['Feature name'], data = feature_name)
# Creates a dataframe with a column titled 'Feature name' and row values contained in the 'feature_name' variable.
summary_table['Coefficients'] = np.transpose(model_loaded.coef_)
# Creates a new column in the dataframe, called 'Coefficients',
# with row values the transposed coefficients from the 'LogisticRegression' object.
summary_table.index = summary_table.index + 1
# Increases the index of every row of the dataframe with 1.
summary_table.loc[0] = ['Intercept', model_loaded.intercept_]
# Assigns values of the row with index 0 of the dataframe.
summary_table = summary_table.sort_index()
# Sorts the dataframe by index.
p_values = model_loaded.p_values
# We take the result of the newly added method 'p_values' and store it in a variable 'p_values'.
p_values = np.append(np.nan,np.array(p_values))
# We add the value 'NaN' in the beginning of the variable with p-values.
summary_table['p_values'] = p_values
# In the 'summary_table' dataframe, we add a new column, called 'p_values', containing the values from the 'p_values' variable.
summary_table
# In the 'summary_table' dataframe, we add a new column, called 'p_values', containing the values from the 'p_values' variable.
save_path = os.path.join(config.SAVE_MODEL_PATH,config.SCORECARD_EAD)

summary_table.to_csv(save_path)

