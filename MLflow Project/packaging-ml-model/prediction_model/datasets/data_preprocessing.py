import sys
import os
import joblib
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

PACKAGE_ROOT = Path(os.path.abspath(os.path.dirname(__file__))).parent.parent
sys.path.append(str(PACKAGE_ROOT))

from prediction_model.config import config
from prediction_model.processing.data_handling import load_dataset

filepath = os.path.join(config.DATAPATH,'loan_data_2007_2014.csv')

loan_data = pd.read_csv(filepath)

#loan_data = loan_data_backup.copy()

# Replace specific text patterns with appropriate numeric values
loan_data['emp_length_int'] = loan_data['emp_length'].str.replace('10+ years', '10')
loan_data['emp_length_int'] = loan_data['emp_length_int'].str.replace('< 1 year', '0')
loan_data['emp_length_int'] = loan_data['emp_length_int'].str.replace('n/a', '0')
loan_data['emp_length_int'] = loan_data['emp_length_int'].str.replace(' years', '')
loan_data['emp_length_int'] = loan_data['emp_length_int'].str.replace(' year', '')

loan_data['emp_length_int'] = pd.to_numeric(loan_data['emp_length_int'])

loan_data['earliest_cr_line_date'] = pd.to_datetime(loan_data['earliest_cr_line'], format = '%b-%y')

average_days_in_month = 30.44
loan_data['mths_since_earliest_cr_line'] = (((pd.to_datetime('2017-12-01') - loan_data['earliest_cr_line_date']).dt.days  / average_days_in_month).fillna(0)).round()

loan_data.loc[: , ['earliest_cr_line', 'earliest_cr_line_date', 'mths_since_earliest_cr_line']][loan_data['mths_since_earliest_cr_line'] < 0]

loan_data['mths_since_earliest_cr_line'][loan_data['mths_since_earliest_cr_line'] < 0] = loan_data['mths_since_earliest_cr_line'].max()

loan_data['term_int'] = loan_data['term'].str.replace(' months', '')

loan_data['term_int'] = pd.to_numeric(loan_data['term'].str.replace(' months', ''))

loan_data['issue_d'] = pd.to_datetime(loan_data['issue_d'], format = '%b-%y')

average_days_in_month = 30.44
loan_data['mths_since_issue_d'] = (((pd.to_datetime('2017-12-01') - loan_data['issue_d']).dt.days  / average_days_in_month).fillna(0)).round()

loan_data_dummies = [pd.get_dummies(loan_data['grade'], prefix = 'grade', prefix_sep = ':').astype(np.int32),
                     pd.get_dummies(loan_data['sub_grade'], prefix = 'sub_grade', prefix_sep = ':').astype(np.int32),
                     pd.get_dummies(loan_data['home_ownership'], prefix = 'home_ownership', prefix_sep = ':').astype(np.int32),
                     pd.get_dummies(loan_data['verification_status'], prefix = 'verification_status', prefix_sep = ':').astype(np.int32),
                     pd.get_dummies(loan_data['loan_status'], prefix = 'loan_status', prefix_sep = ':').astype(np.int32),
                     pd.get_dummies(loan_data['purpose'], prefix = 'purpose', prefix_sep = ':').astype(np.int32),
                     pd.get_dummies(loan_data['addr_state'], prefix = 'addr_state', prefix_sep = ':').astype(np.int32),
                     pd.get_dummies(loan_data['initial_list_status'], prefix = 'initial_list_status', prefix_sep = ':').astype(np.int32)]


loan_data_dummies = pd.concat(loan_data_dummies, axis = 1)

loan_data = pd.concat([loan_data, loan_data_dummies], axis = 1)

loan_data['total_rev_hi_lim'].fillna(loan_data['funded_amnt'], inplace = True)

loan_data['annual_inc'].fillna(loan_data['annual_inc'].mean(), inplace=True)

loan_data['mths_since_earliest_cr_line'].fillna(0, inplace=True)
loan_data['acc_now_delinq'].fillna(0, inplace=True)
loan_data['total_acc'].fillna(0, inplace=True)
loan_data['pub_rec'].fillna(0, inplace=True)
loan_data['open_acc'].fillna(0, inplace=True)
loan_data['inq_last_6mths'].fillna(0, inplace=True)
loan_data['delinq_2yrs'].fillna(0, inplace=True)
loan_data['emp_length_int'].fillna(0, inplace=True)



loan_data['good_bad'] = np.where(loan_data['loan_status'].isin(['Charged Off', 'Default',
                                                               'Does not meet the credit policy. Status:Charged Off',
                                                               'Late (31-120 days)']), 0, 1)

preprocessed_file = os.path.join(config.DATAPATH,'loan_data_2007_2014_preprocessed.csv')

print(len(loan_data.columns))

loan_data.to_csv(preprocessed_file)

# Here we take only the accounts that were charged-off (written-off).
loan_data_defaults = loan_data[loan_data['loan_status'].isin(['Charged Off','Does not meet the credit policy. Status:Charged Off'])]

loan_data_defaults['mths_since_last_delinq'].fillna(0, inplace = True)
loan_data_defaults['mths_since_last_record'].fillna(0, inplace=True)

# We calculate the dependent variable for the LGD model: recovery rate.
# It is the ratio of recoveries and funded amount.
loan_data_defaults['recovery_rate'] = loan_data_defaults['recoveries'] / loan_data_defaults['funded_amnt']

# We set recovery rates that are greater than 1 to 1 and recovery rates that are less than 0 to 0.
loan_data_defaults['recovery_rate'] = np.where(loan_data_defaults['recovery_rate'] > 1, 1, loan_data_defaults['recovery_rate'])
loan_data_defaults['recovery_rate'] = np.where(loan_data_defaults['recovery_rate'] < 0, 0, loan_data_defaults['recovery_rate'])

# We calculate the dependent variable for the EAD model: credit conversion factor.
# It is the ratio of the difference of the amount used at the moment of default to the total funded amount.
loan_data_defaults['CCF'] = (loan_data_defaults['funded_amnt'] - loan_data_defaults['total_rec_prncp']) / loan_data_defaults['funded_amnt']

# We create a new variable which is 0 if recovery rate is 0 and 1 otherwise.
loan_data_defaults['recovery_rate_0_1'] = np.where(loan_data_defaults['recovery_rate'] == 0, 0, 1)

preprocessed_file = os.path.join(config.DATAPATH,'loan_data_defaults.csv')

loan_data_defaults.to_csv(preprocessed_file)



#############################--------loan_data_2015----------#######################################

filepath = os.path.join(config.DATAPATH,'loan_data_2015.csv')

loan_data = pd.read_csv(filepath)

#loan_data = loan_data_backup.copy()

# Replace specific text patterns with appropriate numeric values
loan_data['emp_length_int'] = loan_data['emp_length'].str.replace('10+ years', '10')
loan_data['emp_length_int'] = loan_data['emp_length_int'].str.replace('< 1 year', '0')
loan_data['emp_length_int'] = loan_data['emp_length_int'].str.replace('n/a', '0')
loan_data['emp_length_int'] = loan_data['emp_length_int'].str.replace(' years', '')
loan_data['emp_length_int'] = loan_data['emp_length_int'].str.replace(' year', '')

loan_data['emp_length_int'] = pd.to_numeric(loan_data['emp_length_int'])

loan_data['earliest_cr_line_date'] = pd.to_datetime(loan_data['earliest_cr_line'], format = '%b-%y')

average_days_in_month = 30.44
loan_data['mths_since_earliest_cr_line'] = (((pd.to_datetime('2017-12-01') - loan_data['earliest_cr_line_date']).dt.days  / average_days_in_month).fillna(0)).round()

loan_data.loc[: , ['earliest_cr_line', 'earliest_cr_line_date', 'mths_since_earliest_cr_line']][loan_data['mths_since_earliest_cr_line'] < 0]

loan_data['mths_since_earliest_cr_line'][loan_data['mths_since_earliest_cr_line'] < 0] = loan_data['mths_since_earliest_cr_line'].max()

loan_data['term_int'] = loan_data['term'].str.replace(' months', '')

loan_data['term_int'] = pd.to_numeric(loan_data['term'].str.replace(' months', ''))

loan_data['issue_d'] = pd.to_datetime(loan_data['issue_d'], format = '%b-%y')

average_days_in_month = 30.44
loan_data['mths_since_issue_d'] = (((pd.to_datetime('2017-12-01') - loan_data['issue_d']).dt.days  / average_days_in_month).fillna(0)).round()

loan_data_dummies = [pd.get_dummies(loan_data['grade'], prefix = 'grade', prefix_sep = ':').astype(np.int32),
                     pd.get_dummies(loan_data['sub_grade'], prefix = 'sub_grade', prefix_sep = ':').astype(np.int32),
                     pd.get_dummies(loan_data['home_ownership'], prefix = 'home_ownership', prefix_sep = ':').astype(np.int32),
                     pd.get_dummies(loan_data['verification_status'], prefix = 'verification_status', prefix_sep = ':').astype(np.int32),
                     pd.get_dummies(loan_data['loan_status'], prefix = 'loan_status', prefix_sep = ':').astype(np.int32),
                     pd.get_dummies(loan_data['purpose'], prefix = 'purpose', prefix_sep = ':').astype(np.int32),
                     pd.get_dummies(loan_data['addr_state'], prefix = 'addr_state', prefix_sep = ':').astype(np.int32),
                     pd.get_dummies(loan_data['initial_list_status'], prefix = 'initial_list_status', prefix_sep = ':').astype(np.int32)]


loan_data_dummies = pd.concat(loan_data_dummies, axis = 1)

loan_data = pd.concat([loan_data, loan_data_dummies], axis = 1)

loan_data['total_rev_hi_lim'].fillna(loan_data['funded_amnt'], inplace = True)

loan_data['annual_inc'].fillna(loan_data['annual_inc'].mean(), inplace=True)

loan_data['mths_since_earliest_cr_line'].fillna(0, inplace=True)
loan_data['acc_now_delinq'].fillna(0, inplace=True)
loan_data['total_acc'].fillna(0, inplace=True)
loan_data['pub_rec'].fillna(0, inplace=True)
loan_data['open_acc'].fillna(0, inplace=True)
loan_data['inq_last_6mths'].fillna(0, inplace=True)
loan_data['delinq_2yrs'].fillna(0, inplace=True)
loan_data['emp_length_int'].fillna(0, inplace=True)



loan_data['good_bad'] = np.where(loan_data['loan_status'].isin(['Charged Off', 'Default',
                                                               'Does not meet the credit policy. Status:Charged Off',
                                                               'Late (31-120 days)']), 0, 1)

preprocessed_file = os.path.join(config.DATAPATH,'loan_data_2015_preprocessed.csv')

loan_data.to_csv(preprocessed_file)














