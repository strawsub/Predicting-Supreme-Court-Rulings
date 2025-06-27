import pickle
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler

from utils import data_loader
from utils import learning_curve

def ensemble_learning(train_data, dev_data):
    # Prepare training features and labels
    X_train, y_train = train_data.drop('successful_appeal', axis=1), train_data['successful_appeal']
    X_dev, y_dev = dev_data.drop('successful_appeal', axis=1), dev_data['successful_appeal']

    # Scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_dev_scaled = scaler.transform(X_dev[X_train.columns])

    print(X_train_scaled.shape)
    print(X_dev_scaled.shape)

    # Create individual classifiers
    model_files = [
        '../stacking_pre_models/decision_tree_model.pkl',
        '../stacking_pre_models/knn_model.pkl',
        '../stacking_pre_models/logistic_regression_model.pkl',
        '../stacking_pre_models/naive_bayes_model.pkl',
        '../stacking_pre_models/random_forest_model.pkl'
    ]

    models = {}
    for file in model_files:
        with open(file, 'rb') as f:
            model_name = file.split('/')[-1].split('_model')[0]
            models[model_name] = pickle.load(f)

    with open('ensemble_complex_model.pkl', 'rb') as f:
        ensemble_model = pickle.load(f)

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

    # Function to analyze overlapping predictions
    def analyze_overlapping_predictions(models, X_dev):
        # Generate predictions for all models
        predictions = {name: model.predict(X_dev) for name, model in models.items()}
        predictions_df = pd.DataFrame(predictions)

        # Calculate overlaps between all pairs of models
        overlap_matrix = pd.DataFrame(index=models.keys(), columns=models.keys())
        for model1 in models.keys():
            for model2 in models.keys():
                overlap = (predictions_df[model1] == predictions_df[model2]).sum()
                overlap_matrix.loc[model1, model2] = overlap

        print("Overlapping Predictions Matrix:")
        print(overlap_matrix)

        prediction_counts = predictions_df.apply(pd.Series.value_counts).fillna(0).astype(int)
        print("Class Predictions Count for Each Model:")
        print(prediction_counts)

        # Display the matrix using seaborn heatmap
        plt.figure(figsize=(12, 10))
        sns.heatmap(overlap_matrix.astype(int), annot=True, cmap='coolwarm', fmt='d', annot_kws={"size": 16}, cbar=True)
        plt.title('Overlapping Predictions Heatmap')
        plt.tight_layout()
        plt.show()

    # Analyze overlapping predictions among the base models
    analyze_overlapping_predictions(models, X_dev_scaled)

if __name__ == "__main__":
    # Pre-hearing dataset
    train_df_pre, encoder_pre = data_loader.load_train_encode_pre()
    dev_df_pre = data_loader.load_dev_encode_pre(encoder_pre)
    ensemble_learning(train_df_pre, dev_df_pre)