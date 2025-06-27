import pickle
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from utils import data_loader

def knn_model(train_df, target_column='successful_appeal'):
    """
    Perform hyperparameter tuning for K-Nearest Neighbors (KNN) using GridSearchCV.

    Parameters:
    - train_df: Training dataset as a pandas DataFrame
    - target_column: The column name of the target variable (default is 'successful_appeal')

    Returns:
    - best_model: The KNN model with the best hyperparameters
    - best_params: The best hyperparameters found by GridSearchCV
    """
    # Split features and target for training dataset
    X_train = train_df.drop(target_column, axis=1)
    y_train = train_df[target_column]

    # Define the parameter grid for hyperparameter tuning
    param_grid = {
        'n_neighbors': [10, 30, 50],  # Number of neighbors to use
        'weights': ['uniform', 'distance'],  # Weight function used in prediction
        'metric': ['manhattan', 'manhattan']  # Distance metric
    }

    # Initialize the KNeighborsClassifier
    knn_model = KNeighborsClassifier()

    # Set up the GridSearchCV for hyperparameter tuning
    grid_search = GridSearchCV(estimator=knn_model, param_grid=param_grid, cv=5, scoring='f1_weighted', n_jobs=-1, verbose=2)

    # Fit GridSearchCV
    grid_search.fit(X_train, y_train)

    # Get the best model and parameters from grid search
    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_

    print(f"Best Parameters: {best_params}")

    return best_model

# Example usage:
if __name__ == "__main__":
    # Load your datasets using your data_loader functions
    train_df, encoder = data_loader.load_train_encode_pre()
    # Best Parameters: {'metric': 'manhattan', 'n_neighbors': 11, 'weights': 'uniform'}
    best_model = knn_model(train_df)

    with open('knn_model.pkl', 'wb') as f:
        pickle.dump(best_model, f)