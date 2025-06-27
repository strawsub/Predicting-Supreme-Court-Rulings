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

def logistic_regression(train_data, test_data):
    # Prepare training features and labels
    X_train, y_train = train_data.drop('successful_appeal', axis=1), train_data['successful_appeal']
    X_test, case_ids = test_data, pd.read_json('../data/test.jsonl', lines=True)['case_id']
    X_test.fillna(0, inplace=True)
    # Scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test.reindex(columns=X_train.columns, fill_value=0))


    # Initialize the base models
    with open('../stacking_post_models/logistic_regression_model.pkl', 'rb') as f:
        model = pickle.load(f)

    # Make predictions on the test set
    y_test_pred = model.predict(X_test_scaled)

    # Create a DataFrame with case IDs and predictions
    results = pd.DataFrame({
        'case_id': case_ids,
        'successful_appeal': y_test_pred
    })

    # Export predictions to a CSV file
    results_filename = 'logistic_test.csv'
    results.to_csv(results_filename, index=False)

    print(f"Predictions exported as {results_filename}")

    # Save the trained ensemble model as a pickle file
    model_filename = 'logistic_test.pkl'
    with open(model_filename, 'wb') as model_file:
        pickle.dump(model, model_file)

    print(f"Model saved as {model_filename}")

if __name__ == "__main__":
    train_df, encoder = data_loader.load_train_encode_full()
    test_df = data_loader.load_test_encode_full(encoder)
    logistic_regression(train_df, test_df)