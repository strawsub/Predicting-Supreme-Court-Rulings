import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import VotingClassifier, RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import learning_curve
from utils import data_loader
from utils import learning_curve

def ensemble_learning(train_data, dev_data):
    # Prepare training features and labels
    X_train, y_train = train_data.drop('successful_appeal', axis=1), train_data['successful_appeal']
    X_dev, y_dev = dev_data.drop('successful_appeal', axis=1), dev_data['successful_appeal']

    # Scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_dev_scaled = scaler.transform(X_dev.reindex(columns=X_train.columns, fill_value=0))

    # Create individual classifiers
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
        # ('logistic', logistic_model),
        # ('naive_bayes', naive_bayes_model),
        ('decision_tree', decision_tree_model),
        ('knn', knn_model),
        ('random_forest', random_forest_model)
    ], voting='hard')

    # Plot learning curve
    learning_curve.plot_learning_curve_cv(ensemble_model, train_data, "Learning Curve for Ensemble Model (3 base models & tuned)")

    with open('ensemble_complex_model.pkl', 'wb') as f:
        pickle.dump(ensemble_model, f)

    # Fit the ensemble model
    ensemble_model.fit(X_train_scaled, y_train)

    # Make predictions on the dev set
    y_dev_pred = ensemble_model.predict(X_dev_scaled)

    # Print classification report for development set
    print("Classification Report on Development Set:")
    print(classification_report(y_dev, y_dev_pred))

    # Compute confusion matrix for development set
    cm = confusion_matrix(y_dev, y_dev_pred)

    # Print confusion matrix
    print("Confusion Matrix for Development Set:")
    print(cm)

if __name__ == "__main__":
    # Pre-hearing dataset
    train_df_pre, encoder_pre = data_loader.load_train_encode_pre()
    dev_df_pre = data_loader.load_dev_encode_pre(encoder_pre)
    ensemble_learning(train_df_pre, dev_df_pre)