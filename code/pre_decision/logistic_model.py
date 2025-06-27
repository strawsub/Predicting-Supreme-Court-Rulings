import pickle

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
from utils import data_loader
from utils import learning_curve

def logistic_hyperparameter_tuning(train_set, dev_set):
    """
    Perform hyperparameter tuning for Logistic Regression using GridSearchCV.

    Parameters:
    - train_set: Training dataset (DataFrame)
    - dev_set: Development dataset (DataFrame)

    Returns:
    - best_model: The Logistic Regression model with the best hyperparameters
    - best_params: The best hyperparameters found by GridSearchCV
    """
    # Separate features and labels
    X_train = train_set.drop('successful_appeal', axis=1)
    y_train = train_set['successful_appeal']
    X_dev = dev_set.drop('successful_appeal', axis=1)
    y_dev = dev_set['successful_appeal']

    # Scale the data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_dev_scaled = scaler.transform(X_dev)

    # Define hyperparameters to tune
    param_grid = {
        'C': [0.0001, 0.001, 0.01, 0.1, 1],  # Regularization strength
        'solver': ['lbfgs', 'liblinear'],
        'max_iter': [300, 500, 700],  # Max number of iterations
    }

    # Logistic Regression model
    lr_model = LogisticRegression(max_iter=1000)

    # GridSearchCV for hyperparameter tuning
    grid_search = GridSearchCV(estimator=lr_model, param_grid=param_grid, cv=5, scoring='f1_weighted', n_jobs=-1)
    grid_search.fit(X_train_scaled, y_train)

    # Get the best model and parameters
    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_
    print("Best parameters: ", best_params)
    return best_model, best_params


def best_logistic_classifier(train_set, dev_set, best_params=None):
    """
    Train Logistic Regression using the best hyperparameters and evaluate it.

    Parameters:
    - train_set: Training dataset (DataFrame)
    - dev_set: Development dataset (DataFrame)
    - best_params: Best hyperparameters from tuning

    Returns:
    - logistic_model: Trained Logistic Regression model
    """
    # Separate features and labels
    X_train = train_set.drop('successful_appeal', axis=1)
    y_train = train_set['successful_appeal']
    X_dev = dev_set.drop('successful_appeal', axis=1)
    y_dev = dev_set['successful_appeal']

    # Scale the data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_dev_scaled = scaler.transform(X_dev)
    print(X_dev_scaled.shape)

    # If no best_params provided, use default values
    if best_params is None:
        best_params = {'C': 0.1, 'max_iter': 300, 'solver': 'lbfgs'}


    # Train logistic regression with the best parameters
    logistic_model = LogisticRegression(**best_params)
    logistic_model.fit(X_train_scaled, y_train)

    # Evaluation on the development set
    print("Classification Report on Development Set:")
    y_dev_pred = logistic_model.predict(X_dev_scaled)
    print(classification_report(y_dev, y_dev_pred))

    # Feature importance score
    importance = np.abs(logistic_model.coef_[0])

    # Create a dataframe to view the features and their importance scores
    feature_importance_df = pd.DataFrame({
        'feature': X_train.columns,
        'importance': importance
    }).sort_values(by='importance', ascending=False)
    feature_importance_df.to_csv("logistic_feature.csv", index=False)

    return logistic_model


if __name__ == "__main__":
    # Pre-hearing dataset
    print("-------------------------------------------------")
    print("Pre-hearing dataset")
    train_df_pre, encoder_pre = data_loader.load_train_encode_pre()
    dev_df_pre = data_loader.load_dev_encode_pre(encoder_pre)
    best_model_pre, best_params_pre = logistic_hyperparameter_tuning(train_df_pre, dev_df_pre)
    learning_curve.plot_learning_curve_cv(best_model_pre, train_df_pre,"Learning Curve of Logistic Regression for Pre-Decision Dataset")
    with open('../stacking_pre_models/logistic_regression_model.pkl', 'wb') as f:
        pickle.dump(best_model_pre, f)
    logistic_model = best_logistic_classifier(train_df_pre, dev_df_pre, best_params_pre)

    # Post-hearing dataset
    print("-------------------------------------------------")
    print("Post-hearing dataset")
    train_df_post, encoder_post = data_loader.load_train_encode_post()
    dev_df_post = data_loader.load_dev_encode_post(encoder_post)
    best_model_post, best_params_post = logistic_hyperparameter_tuning(train_df_post, dev_df_post)
    learning_curve.plot_learning_curve_cv(best_model_post, train_df_post,"Learning Curve of Logistic Regression for Post-Decision Dataset")
    with open('../stacking_post_models/logistic_regression_model.pkl', 'wb') as f:
        pickle.dump(best_model_post, f)
    logistic_model = best_logistic_classifier(train_df_post, dev_df_post, best_params_post)
