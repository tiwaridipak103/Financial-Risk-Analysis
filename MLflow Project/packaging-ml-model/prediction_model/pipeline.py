import os
import pandas as pd
import sys
import joblib
from pathlib import Path

PACKAGE_ROOT = Path(os.path.abspath(os.path.dirname(__file__))).parent
sys.path.append(str(PACKAGE_ROOT))


from prediction_model.config import config
from sklearn.pipeline import Pipeline
from prediction_model.config import config
import prediction_model.processing.preprocessing as pp 
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn import linear_model
import scipy.stats as stat

#Build a Logistic Regression Model with P-Values
class LogisticRegression_with_p_values:
    
    def __init__(self,*args,**kwargs):
        self.model = linear_model.LogisticRegression(*args,**kwargs)

    def fit(self,X,y):
        self.model.fit(X,y)

        #### Get p-values for the fitted model ####
        denom = (2.0 * (1.0 + np.cosh(self.model.decision_function(X))))
        denom = np.tile(denom,(X.shape[1],1)).T
        F_ij = np.dot((X / denom).T,X)
        Cramer_Rao = np.linalg.inv(F_ij)
        sigma_estimates = np.sqrt(np.diagonal(Cramer_Rao))
        z_scores = self.model.coef_[0] / sigma_estimates
        p_values = [stat.norm.sf(abs(x)) * 2 for x in z_scores]
        self.coef_ = self.model.coef_
        self.intercept_ = self.model.intercept_
        self.p_values = p_values



class LinearRegression(linear_model.LinearRegression):
    def __init__(self, fit_intercept=True, normalize=False, copy_X=True,
                 n_jobs=1):
        self.fit_intercept = fit_intercept
        self.normalize = normalize
        self.copy_X = copy_X
        self.n_jobs = n_jobs
    def fit(self, X, y, n_jobs=1):
        self = super(LinearRegression, self).fit(X, y, n_jobs)
        sse = np.sum((self.predict(X) - y) ** 2, axis=0) / float(X.shape[0] - X.shape[1])
        se = np.array([np.sqrt(np.diagonal(sse * np.linalg.inv(np.dot(X.T, X))))])
        self.t = self.coef_ / se
        self.p = np.squeeze(2 * (1 - stat.t.cdf(np.abs(self.t), y.shape[0] - X.shape[1])))
        return self


classification_pipeline = Pipeline(
    [
        ('One_to_One', pp.One_to_One(variables=config.FEATURES_SELECTION)),
        ('LogisticClassifier',LogisticRegression_with_p_values(random_state=0))
    ]
)

linear_pipeline = Pipeline(
    [
        ('One_to_One', pp.One_to_One(variables=config.FEATURES_SELECTION)),
        ('LogisticClassifier',LogisticRegression_with_p_values(random_state=0))
    ]
)

# classification_pipeline = Pipeline(
#     [
#         ('DomainProcessing',pp.DomainProcessing(variable_to_modify = config.FEATURE_TO_MODIFY,
#         variable_to_add = config.FEATURE_TO_ADD)),
#         ('MeanImputation', pp.MeanImputer(variables=config.NUM_FEATURES)),
#         ('ModeImputation',pp.ModeImputer(variables=config.CAT_FEATURES)),
#         ('DropFeatures', pp.DropColumns(variables_to_drop=config.DROP_FEATURES)),
#         ('LabelEncoder',pp.CustomLabelEncoder(variables=config.FEATURES_TO_ENCODE)),
#         ('LogTransform',pp.LogTransforms(variables=config.LOG_FEATURES)),
#         ('MinMaxScale', MinMaxScaler()),
#         ('LogisticClassifier',LogisticRegression(random_state=0))
#     ]
# )