import pickle

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from utils import data_loader
import contextlib

output_file = 'active_learning_model.txt'

def active_learning(train_data, dev_data, classifier, n_initial=800, n_cycles=20, n_samples=20):
    # Prepare features and labels
    X_train = train_data.drop(columns=['successful_appeal'])
    y_train = train_data['successful_appeal'].values

    # Prepare development features and labels
    X_dev = dev_data.drop(columns=['successful_appeal'])
    y_dev = dev_data['successful_appeal'].values

    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_dev_scaled = scaler.transform(X_dev)

    # Create a mask for labeled data
    labeled_mask = np.zeros(y_train.shape, dtype=bool)
    labeled_mask[:n_initial] = True

    # Active learning loop
    for cycle in range(n_cycles):
        # Break if all samples are labeled
        if np.all(labeled_mask):
            print(f"All samples labeled after {cycle} cycles.")
            break

        # Fit the model on the currently labeled data
        clf = classifier
        clf.fit(X_train_scaled[labeled_mask], y_train[labeled_mask])

        # Predict probabilities for the unlabeled instances
        y_proba = clf.predict_proba(X_train_scaled[~labeled_mask])

        # Calculate uncertainty (using entropy)
        uncertainty = -np.sum(y_proba * np.log2(y_proba + 1e-10), axis=1)

        # Select the indices of the most uncertain instances
        query_indices = np.argsort(-uncertainty)[:n_samples]  # Select the top n_samples most uncertain

        # Update the labeled mask with the selected indices
        labeled_mask[~labeled_mask][query_indices] = True

    # Final model evaluation on the labeled dataset
    clf.fit(X_train_scaled[labeled_mask], y_train[labeled_mask])
    y_pred = clf.predict(X_dev_scaled)

    # Print classification report
    print("Classification Report:")
    print(classification_report(y_dev, y_pred))

    # Print the total number of samples used
    print(f"Total number of samples used: {np.sum(labeled_mask)}")

if __name__ == "__main__":
    with open(output_file, 'w') as f:
        with contextlib.redirect_stdout(f):
            # Load your datasets
            train_data, encoder = data_loader.load_train_encode_pre()
            dev_data = data_loader.load_dev_encode_pre(encoder)

            # Active learning with Logistic Regression
            print("Active Learning with Logistic Regression:")
            with open('../stacking_pre_models/logistic_regression_model.pkl', 'rb') as f:
                logistic_regression_model = pickle.load(f)
            active_learning(train_data, dev_data, logistic_regression_model)

            # Active learning with Decision Tree
            print("\nActive Learning with Decision Tree:")
            with open('../stacking_pre_models/decision_tree_model.pkl', 'rb') as f:
                decision_tree_model = pickle.load(f)
            active_learning(train_data, dev_data, decision_tree_model)

            # Active learning with Naive Bayes
            print("\nActive Learning with Naive Bayes:")
            with open('../stacking_pre_models/naive_bayes_model.pkl', 'rb') as f:
                naive_bayes_model = pickle.load(f)
            active_learning(train_data, dev_data, naive_bayes_model)

            # Active learning with KNN
            print("\nActive Learning with KNN:")
            with open('../stacking_pre_models/knn_model.pkl', 'rb') as f:
                knn_model = pickle.load(f)
            active_learning(train_data, dev_data, knn_model)