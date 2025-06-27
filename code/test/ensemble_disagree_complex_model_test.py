import os
import pickle

import numpy as np
import pandas as pd
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix

from utils import data_loader

def identify_disagreement_instances(train_data):
    # Prepare training features and labels
    X_train, y_train = train_data.drop('successful_appeal', axis=1), train_data['successful_appeal']
    scaler = StandardScaler()

    X_train = scaler.fit_transform(X_train)


    with open('../stacking_pre_models/decision_tree_model.pkl', 'rb') as f:
        decision_tree_model = pickle.load(f)
    with open('../stacking_pre_models/knn_model.pkl', 'rb') as f:
        knn_model = pickle.load(f)
    with open('../stacking_pre_models/logistic_regression_model.pkl', 'rb') as f:
        logistic_regression_model = pickle.load(f)
    with open('../stacking_pre_models/naive_bayes_model.pkl', 'rb') as f:
        naive_bayes_model = pickle.load(f)
    with open('../stacking_pre_models/random_forest_model.pkl', 'rb') as f:
        random_forest_model = pickle.load(f)

    # Create an ensemble model using VotingClassifier (majority voting)
    ensemble_model = VotingClassifier(estimators=[
        ('logistic', logistic_regression_model),
        ('gaussian_nb', naive_bayes_model),
        ('decision_tree', decision_tree_model),
        ('random_forest', random_forest_model),
        ('knn', knn_model)
    ], voting='hard')

    # Fit the ensemble and individual classifiers
    decision_tree_model.fit(X_train, y_train)
    knn_model.fit(X_train, y_train)
    logistic_regression_model.fit(X_train, y_train)
    naive_bayes_model.fit(X_train, y_train)
    random_forest_model.fit(X_train, y_train)
    ensemble_model.fit(X_train, y_train)

    # Make predictions with individual classifiers and the ensemble
    pred_clf1 = decision_tree_model.predict(X_train)
    pred_clf2 = knn_model.predict(X_train)
    pred_clf3 = logistic_regression_model.predict(X_train)
    pred_clf4 = naive_bayes_model.predict(X_train)
    pred_clf5 = random_forest_model.predict(X_train)
    ensemble_pred = ensemble_model.predict(X_train)

    # Stack the individual predictions
    individual_preds = np.vstack([pred_clf1, pred_clf2, pred_clf3, pred_clf4, pred_clf5])

    # Count how many classifiers disagree with the ensemble
    disagreements = np.sum(individual_preds != ensemble_pred, axis=0)

    # Identify instances where 2 or more classifiers disagree with the ensemble
    disagreement_indices = np.where(disagreements == 2)[0]

    # Filter the training data to exclude these disagreement instances
    filtered_train_data = train_data.drop(train_data.index[disagreement_indices])

    print(f"Excluded {len(disagreement_indices)} instances where 2 or more classifiers disagreed with the ensemble.")

    return filtered_train_data


def ensemble_learning(train_data, test_data):
    # Prepare training features and labels
    X_train, y_train = train_data.drop('successful_appeal', axis=1), train_data['successful_appeal']
    X_test, case_ids = test_data, pd.read_json('../data/test.jsonl', lines=True)['case_id']
    X_test.fillna(0, inplace=True)
    # Scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test.reindex(columns=X_train.columns, fill_value=0))


    # Initialize the base models
    logistic_model = LogisticRegression(penalty='l2', C=0.1, solver='lbfgs', max_iter=6000)
    decision_tree_model = DecisionTreeClassifier(max_depth=5)
    knn_model = KNeighborsClassifier(n_neighbors=100)
    naive_bayes_model = GaussianNB()
    random_forest_model = RandomForestClassifier(max_depth=20)

    # Create the Voting Classifier (hard voting)
    ensemble_model = VotingClassifier(estimators=[
        ('logistic', logistic_model),
        ('naive_bayes', naive_bayes_model),
        ('decision_tree', decision_tree_model),
        ('knn', knn_model),
        ('random_forest', random_forest_model)
    ], voting='hard')

    # Fit the ensemble model
    ensemble_model.fit(X_train_scaled, y_train)

    # Make predictions on the test set
    y_test_pred = ensemble_model.predict(X_test_scaled)

    # Create a DataFrame with case IDs and predictions
    results = pd.DataFrame({
        'case_id': case_ids,
        'successful_appeal': y_test_pred
    })

    # Export predictions to a CSV file
    results_filename = 'ensemble_complex.csv'
    results.to_csv(results_filename, index=False)

    print(f"Predictions exported as {results_filename}")

    # Save the trained ensemble model as a pickle file
    model_filename = 'ensemble_model.pkl'
    with open(model_filename, 'wb') as model_file:
        pickle.dump(ensemble_model, model_file)

    print(f"Model saved as {model_filename}")

if __name__ == "__main__":
    train_df, encoder = data_loader.load_train_encode_full()
    filtered_train_data = identify_disagreement_instances(train_df)
    test_df = data_loader.load_test_encode_full(encoder)
    ensemble_learning(filtered_train_data, test_df)