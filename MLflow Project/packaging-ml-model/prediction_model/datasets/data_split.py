import sys
import os
import joblib
from pathlib import Path
import numpy as np
import pandas as pd

PACKAGE_ROOT = Path(os.path.abspath(os.path.dirname(__file__))).parent.parent
sys.path.append(str(PACKAGE_ROOT))

from prediction_model.config import config

filepath = os.path.join(config.DATAPATH,'loan_data_2007_2014_preprocessed.csv')

loan_data = pd.read_csv(filepath)

#Splitting Data
from sklearn.model_selection import train_test_split

loan_data_inputs_train, loan_data_inputs_test, loan_data_targets_train, loan_data_targets_test = train_test_split(loan_data.drop('good_bad', axis = 1), loan_data['good_bad'], test_size = 0.2, random_state = 42)

def data_preprocessing(df_input):
    df_input['home_ownership:RENT_OTHER_NONE_ANY'] = sum([df_input['home_ownership:RENT'], df_input['home_ownership:OTHER'],
                                                    df_input['home_ownership:NONE'], df_input['home_ownership:ANY']])

    if ['addr_state:ND'] in df_input.columns.values:
      pass
    else:
      df_input['addr_state:ND'] = 0
    
    df_input['addr_state:ND_NE_IA_NV_FL_HI_AL'] = sum([df_input['addr_state:ND'], df_input['addr_state:NE'],
                                                 df_input['addr_state:IA'], df_input['addr_state:NV'],
                                                 df_input['addr_state:FL'], df_input['addr_state:HI'],
                                                 df_input['addr_state:AL']])
    
    df_input['addr_state:NM_VA'] = sum([df_input['addr_state:NM'], df_input['addr_state:VA']])
    
    df_input['addr_state:OK_TN_MO_LA_MD_NC'] = sum([df_input['addr_state:OK'], df_input['addr_state:TN'],
                                      df_input['addr_state:MO'], df_input['addr_state:LA'],
                                      df_input['addr_state:MD'], df_input['addr_state:NC']])
    
    df_input['addr_state:UT_KY_AZ_NJ'] = sum([df_input['addr_state:UT'], df_input['addr_state:KY'],
                                      df_input['addr_state:AZ'], df_input['addr_state:NJ']])
    
    df_input['addr_state:AR_MI_PA_OH_MN'] = sum([df_input['addr_state:AR'], df_input['addr_state:MI'],
                                      df_input['addr_state:PA'], df_input['addr_state:OH'],
                                      df_input['addr_state:MN']])
    
    df_input['addr_state:RI_MA_DE_SD_IN'] = sum([df_input['addr_state:RI'], df_input['addr_state:MA'],
                                      df_input['addr_state:DE'], df_input['addr_state:SD'],
                                      df_input['addr_state:IN']])
    
    df_input['addr_state:GA_WA_OR'] = sum([df_input['addr_state:GA'], df_input['addr_state:WA'],
                                      df_input['addr_state:OR']])
    
    df_input['addr_state:WI_MT'] = sum([df_input['addr_state:WI'], df_input['addr_state:MT']])
    
    df_input['addr_state:IL_CT'] = sum([df_input['addr_state:IL'], df_input['addr_state:CT']])
    
    df_input['addr_state:KS_SC_CO_VT_AK_MS'] = sum([df_input['addr_state:KS'], df_input['addr_state:SC'],
                                      df_input['addr_state:CO'], df_input['addr_state:VT'],
                                      df_input['addr_state:AK'], df_input['addr_state:MS']])
    
    df_input['addr_state:WV_NH_WY_DC_ME_ID'] = sum([df_input['addr_state:WV'], df_input['addr_state:NH'],
                                      df_input['addr_state:WY'], df_input['addr_state:DC'],
                                      df_input['addr_state:ME'], df_input['addr_state:ID']])
    
    df_input['purpose:educ__sm_b__wedd__ren_en__mov__house'] = sum([df_input['purpose:educational'],df_input['purpose:small_business'],
                                                         df_input['purpose:wedding'], df_input['purpose:renewable_energy'],
                                                         df_input['purpose:moving'], df_input['purpose:house']])
    df_input['purpose:oth__med__vacation'] = sum([df_input['purpose:other'], df_input['purpose:medical'],
                                     df_input['purpose:vacation']])
    df_input['purpose:major_purch__car__home_impr'] = sum([df_input['purpose:major_purchase'], df_input['purpose:car'],
                                                df_input['purpose:home_improvement']])
    
    df_input['term:36'] = np.where((df_input['term_int'] == 36), 1, 0)
    df_input['term:60'] = np.where((df_input['term_int'] == 60), 1, 0)
    
    df_input['emp_length:0'] = np.where(df_input['emp_length_int'].isin([0]), 1, 0)
    df_input['emp_length:1'] = np.where(df_input['emp_length_int'].isin([1]), 1, 0)
    df_input['emp_length:2-4'] = np.where(df_input['emp_length_int'].isin(range(2, 5)), 1, 0)
    df_input['emp_length:5-6'] = np.where(df_input['emp_length_int'].isin(range(5, 7)), 1, 0)
    df_input['emp_length:7-9'] = np.where(df_input['emp_length_int'].isin(range(7, 10)), 1, 0)
    df_input['emp_length:10'] = np.where(df_input['emp_length_int'].isin([10]), 1, 0)
    
    df_input['mths_since_issue_d:<38'] = np.where(df_input['mths_since_issue_d'].isin(range(38)), 1, 0)
    df_input['mths_since_issue_d:38-39'] = np.where(df_input['mths_since_issue_d'].isin(range(38, 40)), 1, 0)
    df_input['mths_since_issue_d:40-41'] = np.where(df_input['mths_since_issue_d'].isin(range(40, 42)), 1, 0)
    df_input['mths_since_issue_d:42-48'] = np.where(df_input['mths_since_issue_d'].isin(range(42, 49)), 1, 0)
    df_input['mths_since_issue_d:49-52'] = np.where(df_input['mths_since_issue_d'].isin(range(49, 53)), 1, 0)
    df_input['mths_since_issue_d:53-64'] = np.where(df_input['mths_since_issue_d'].isin(range(53, 65)), 1, 0)
    df_input['mths_since_issue_d:65-84'] = np.where(df_input['mths_since_issue_d'].isin(range(65, 85)), 1, 0)
    df_input['mths_since_issue_d:>84'] = np.where(df_input['mths_since_issue_d'].isin(range(85, int(df_input['mths_since_issue_d'].max()))), 1, 0)
    
    df_input['int_rate:<9.548'] = np.where((df_input['int_rate'] <= 9.548), 1, 0)
    df_input['int_rate:9.548-12.025'] = np.where((df_input['int_rate'] > 9.548) & (df_input['int_rate'] <= 12.025), 1, 0)
    df_input['int_rate:12.025-15.74'] = np.where((df_input['int_rate'] > 12.025) & (df_input['int_rate'] <= 15.74), 1, 0)
    df_input['int_rate:15.74-20.281'] = np.where((df_input['int_rate'] > 15.74) & (df_input['int_rate'] <= 20.281), 1, 0)
    df_input['int_rate:>20.281'] = np.where((df_input['int_rate'] > 20.281), 1, 0)
    
    df_input['mths_since_earliest_cr_line:<140'] = np.where(df_input['mths_since_earliest_cr_line'].isin(range(140)), 1, 0)
    df_input['mths_since_earliest_cr_line:141-164'] = np.where(df_input['mths_since_earliest_cr_line'].isin(range(140, 165)), 1, 0)
    df_input['mths_since_earliest_cr_line:165-247'] = np.where(df_input['mths_since_earliest_cr_line'].isin(range(165, 248)), 1, 0)
    df_input['mths_since_earliest_cr_line:248-270'] = np.where(df_input['mths_since_earliest_cr_line'].isin(range(248, 271)), 1, 0)
    df_input['mths_since_earliest_cr_line:271-352'] = np.where(df_input['mths_since_earliest_cr_line'].isin(range(271, 353)), 1, 0)
    df_input['mths_since_earliest_cr_line:>352'] = np.where(df_input['mths_since_earliest_cr_line'].isin(range(353, int(df_input['mths_since_earliest_cr_line'].max()))), 1, 0)

    
    df_input['delinq_2yrs:0'] = np.where((df_input['delinq_2yrs'] == 0), 1, 0)
    df_input['delinq_2yrs:1-3'] = np.where(((df_input['delinq_2yrs'] >= 1) & (df_input['delinq_2yrs'] <= 3)), 1, 0)
    df_input['delinq_2yrs:>=4'] = np.where((df_input['delinq_2yrs'] >= 9), 1, 0)
    
    df_input['inq_last_6mths:0'] = np.where((df_input['inq_last_6mths'] == 0), 1, 0)
    df_input['inq_last_6mths:1-2'] = np.where((df_input['inq_last_6mths'] >= 1) & (df_input['inq_last_6mths'] <= 2), 1, 0)
    df_input['inq_last_6mths:3-6'] = np.where((df_input['inq_last_6mths'] >= 3) & (df_input['inq_last_6mths'] <= 6), 1, 0)
    df_input['inq_last_6mths:>6'] = np.where((df_input['inq_last_6mths'] > 6), 1, 0)
    
    df_input['open_acc:0'] = np.where((df_input['open_acc'] == 0), 1, 0)
    df_input['open_acc:1-3'] = np.where((df_input['open_acc'] >= 1) & (df_input['open_acc'] <= 3), 1, 0)
    df_input['open_acc:4-12'] = np.where((df_input['open_acc'] >= 4) & (df_input['open_acc'] <= 12), 1, 0)
    df_input['open_acc:13-17'] = np.where((df_input['open_acc'] >= 13) & (df_input['open_acc'] <= 17), 1, 0)
    df_input['open_acc:18-22'] = np.where((df_input['open_acc'] >= 18) & (df_input['open_acc'] <= 22), 1, 0)
    df_input['open_acc:23-25'] = np.where((df_input['open_acc'] >= 23) & (df_input['open_acc'] <= 25), 1, 0)
    df_input['open_acc:26-30'] = np.where((df_input['open_acc'] >= 26) & (df_input['open_acc'] <= 30), 1, 0)
    df_input['open_acc:>=31'] = np.where((df_input['open_acc'] >= 31), 1, 0)
    
    df_input['pub_rec:0-2'] = np.where((df_input['pub_rec'] >= 0) & (df_input['pub_rec'] <= 2), 1, 0)
    df_input['pub_rec:3-4'] = np.where((df_input['pub_rec'] >= 3) & (df_input['pub_rec'] <= 4), 1, 0)
    df_input['pub_rec:>=5'] = np.where((df_input['pub_rec'] >= 5), 1, 0)
    
    df_input['total_acc:<=27'] = np.where((df_input['total_acc'] <= 27), 1, 0)
    df_input['total_acc:28-51'] = np.where((df_input['total_acc'] >= 28) & (df_input['total_acc'] <= 51), 1, 0)
    df_input['total_acc:>=52'] = np.where((df_input['total_acc'] >= 52), 1, 0)
    
    df_input['acc_now_delinq:0'] = np.where((df_input['acc_now_delinq'] == 0), 1, 0)
    df_input['acc_now_delinq:>=1'] = np.where((df_input['acc_now_delinq'] >= 1), 1, 0)
    
    df_input['total_rev_hi_lim:<=5K'] = np.where((df_input['total_rev_hi_lim'] <= 5000), 1, 0)
    df_input['total_rev_hi_lim:5K-10K'] = np.where((df_input['total_rev_hi_lim'] > 5000) & (df_input['total_rev_hi_lim'] <= 10000), 1, 0)
    df_input['total_rev_hi_lim:10K-20K'] = np.where((df_input['total_rev_hi_lim'] > 10000) & (df_input['total_rev_hi_lim'] <= 20000), 1, 0)
    df_input['total_rev_hi_lim:20K-30K'] = np.where((df_input['total_rev_hi_lim'] > 20000) & (df_input['total_rev_hi_lim'] <= 30000), 1, 0)
    df_input['total_rev_hi_lim:30K-40K'] = np.where((df_input['total_rev_hi_lim'] > 30000) & (df_input['total_rev_hi_lim'] <= 40000), 1, 0)
    df_input['total_rev_hi_lim:40K-55K'] = np.where((df_input['total_rev_hi_lim'] > 40000) & (df_input['total_rev_hi_lim'] <= 55000), 1, 0)
    df_input['total_rev_hi_lim:55K-95K'] = np.where((df_input['total_rev_hi_lim'] > 55000) & (df_input['total_rev_hi_lim'] <= 95000), 1, 0)
    df_input['total_rev_hi_lim:>95K'] = np.where((df_input['total_rev_hi_lim'] > 95000), 1, 0)
    
    df_input['annual_inc:<20K'] = np.where((df_input['annual_inc'] <= 20000), 1, 0)
    df_input['annual_inc:20K-30K'] = np.where((df_input['annual_inc'] > 20000) & (df_input['annual_inc'] <= 30000), 1, 0)
    df_input['annual_inc:30K-40K'] = np.where((df_input['annual_inc'] > 30000) & (df_input['annual_inc'] <= 40000), 1, 0)
    df_input['annual_inc:40K-50K'] = np.where((df_input['annual_inc'] > 40000) & (df_input['annual_inc'] <= 50000), 1, 0)
    df_input['annual_inc:50K-60K'] = np.where((df_input['annual_inc'] > 50000) & (df_input['annual_inc'] <= 60000), 1, 0)
    df_input['annual_inc:60K-70K'] = np.where((df_input['annual_inc'] > 60000) & (df_input['annual_inc'] <= 70000), 1, 0)
    df_input['annual_inc:70K-80K'] = np.where((df_input['annual_inc'] > 70000) & (df_input['annual_inc'] <= 80000), 1, 0)
    df_input['annual_inc:80K-90K'] = np.where((df_input['annual_inc'] > 80000) & (df_input['annual_inc'] <= 90000), 1, 0)
    df_input['annual_inc:90K-100K'] = np.where((df_input['annual_inc'] > 90000) & (df_input['annual_inc'] <= 100000), 1, 0)
    df_input['annual_inc:100K-120K'] = np.where((df_input['annual_inc'] > 100000) & (df_input['annual_inc'] <= 120000), 1, 0)
    df_input['annual_inc:120K-140K'] = np.where((df_input['annual_inc'] > 120000) & (df_input['annual_inc'] <= 140000), 1, 0)
    df_input['annual_inc:>140K'] = np.where((df_input['annual_inc'] > 140000), 1, 0)
    
    df_input['mths_since_last_delinq:Missing'] = np.where((df_input['mths_since_last_delinq'].isnull()), 1, 0)
    df_input['mths_since_last_delinq:0-3'] = np.where((df_input['mths_since_last_delinq'] >= 0) & (df_input['mths_since_last_delinq'] <= 3), 1, 0)
    df_input['mths_since_last_delinq:4-30'] = np.where((df_input['mths_since_last_delinq'] >= 4) & (df_input['mths_since_last_delinq'] <= 30), 1, 0)
    df_input['mths_since_last_delinq:31-56'] = np.where((df_input['mths_since_last_delinq'] >= 31) & (df_input['mths_since_last_delinq'] <= 56), 1, 0)
    df_input['mths_since_last_delinq:>=57'] = np.where((df_input['mths_since_last_delinq'] >= 57), 1, 0)
    
    df_input['dti:<=1.4'] = np.where((df_input['dti'] <= 1.4), 1, 0)
    df_input['dti:1.4-3.5'] = np.where((df_input['dti'] > 1.4) & (df_input['dti'] <= 3.5), 1, 0)
    df_input['dti:3.5-7.7'] = np.where((df_input['dti'] > 3.5) & (df_input['dti'] <= 7.7), 1, 0)
    df_input['dti:7.7-10.5'] = np.where((df_input['dti'] > 7.7) & (df_input['dti'] <= 10.5), 1, 0)
    df_input['dti:10.5-16.1'] = np.where((df_input['dti'] > 10.5) & (df_input['dti'] <= 16.1), 1, 0)
    df_input['dti:16.1-20.3'] = np.where((df_input['dti'] > 16.1) & (df_input['dti'] <= 20.3), 1, 0)
    df_input['dti:20.3-21.7'] = np.where((df_input['dti'] > 20.3) & (df_input['dti'] <= 21.7), 1, 0)
    df_input['dti:21.7-22.4'] = np.where((df_input['dti'] > 21.7) & (df_input['dti'] <= 22.4), 1, 0)
    df_input['dti:22.4-35'] = np.where((df_input['dti'] > 22.4) & (df_input['dti'] <= 35), 1, 0)
    df_input['dti:>35'] = np.where((df_input['dti'] > 35), 1, 0)
    
    df_input['mths_since_last_record:Missing'] = np.where((df_input['mths_since_last_record'].isnull()), 1, 0)
    df_input['mths_since_last_record:0-2'] = np.where((df_input['mths_since_last_record'] >= 0) & (df_input['mths_since_last_record'] <= 2), 1, 0)
    df_input['mths_since_last_record:3-20'] = np.where((df_input['mths_since_last_record'] >= 3) & (df_input['mths_since_last_record'] <= 20), 1, 0)
    df_input['mths_since_last_record:21-31'] = np.where((df_input['mths_since_last_record'] >= 21) & (df_input['mths_since_last_record'] <= 31), 1, 0)
    df_input['mths_since_last_record:32-80'] = np.where((df_input['mths_since_last_record'] >= 32) & (df_input['mths_since_last_record'] <= 80), 1, 0)
    df_input['mths_since_last_record:81-86'] = np.where((df_input['mths_since_last_record'] >= 81) & (df_input['mths_since_last_record'] <= 86), 1, 0)
    df_input['mths_since_last_record:>=86'] = np.where((df_input['mths_since_last_record'] >= 86), 1, 0)

    return df_input

loan_data_inputs_train = data_preprocessing(loan_data_inputs_train)
loan_data_inputs_test = data_preprocessing(loan_data_inputs_test)

print(loan_data_inputs_train.shape , loan_data_inputs_test.shape)

loan_data_inputs_train.to_csv(os.path.join(config.DATAPATH,'loan_data_inputs_train.csv'))
loan_data_targets_train.to_csv(os.path.join(config.DATAPATH,'loan_data_targets_train.csv'))
loan_data_inputs_test.to_csv(os.path.join(config.DATAPATH,'loan_data_inputs_test.csv'))
loan_data_targets_test.to_csv(os.path.join(config.DATAPATH,'loan_data_targets_test.csv'))

#######################--------loan_data_2015_preprocessed--------#####################

filepath = os.path.join(config.DATAPATH,'loan_data_2015_preprocessed.csv')

loan_data_2015_preprocessed = pd.read_csv(filepath)

loan_data_2015_preprocessed['home_ownership:OTHER'] = 0
loan_data_2015_preprocessed['home_ownership:NONE'] = 0
loan_data_2015_preprocessed['addr_state:IA'] = 0
loan_data_2015_preprocessed['addr_state:ND'] = 0
loan_data_2015_preprocessed['addr_state:ID'] = 0

loan_data_inputs_train = data_preprocessing(loan_data_2015_preprocessed)

loan_data_inputs_train.to_csv(os.path.join(config.DATAPATH,'loan_data_2015_test.csv'))
