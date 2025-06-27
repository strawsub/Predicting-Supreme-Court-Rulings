import pickle

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler

from pre_decision.logistic_model import logistic_hyperparameter_tuning, best_logistic_classifier
from utils import data_loader
from utils import learning_curve

with open('../stacking_pre_models/logistic_regression_model.pkl', 'rb') as f:
    logistic_regression_model = pickle.load(f)

train_data, _ =data_loader.load_train_encode_pre()
X = train_data.drop('successful_appeal', axis=1)

importance = np.abs(logistic_regression_model.coef_[0])

# Create a dataframe to view the features and their importance scores
feature_importance_df = pd.DataFrame({
    'feature': X.columns,
    'importance': importance
}).sort_values(by='importance', ascending=False)

feature_importance_df.to_csv("logistic_feature_importance.csv", index=False)

train_df_pre, encoder_pre = data_loader.load_train_encode_pre()
train_df_pre = train_df_pre.drop(columns=[col for col in train_df_pre.columns if 'state' in col or 'category' in col or 'date' in col or 'year' in col])
dev_df_pre = data_loader.load_dev_encode_pre(encoder_pre)
dev_df_pre = dev_df_pre.drop(columns=[col for col in dev_df_pre.columns if 'state' in col or 'category' in col or 'date' in col or 'year' in col])
best_model_pre, best_params_pre = logistic_hyperparameter_tuning(train_df_pre, dev_df_pre)
learning_curve.plot_learning_curve_cv(best_model_pre, train_df_pre)
logistic_model = best_logistic_classifier(train_df_pre, dev_df_pre, best_params_pre)