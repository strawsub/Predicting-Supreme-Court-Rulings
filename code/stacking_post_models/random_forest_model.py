import pickle

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from utils import data_loader


def random_forest(train_df, target_column='successful_appeal'):
    """
    Perform hyperparameter tuning for Random Forest using GridSearchCV and evaluate on the development set.

    Parameters:
    - train_df: Training dataset as a pandas DataFrame
    - dev_df: Development (pre_decision) dataset as a pandas DataFrame
    - target_column: The column name of the target variable (default is 'successful_appeal')

    Returns:
    - best_model: The Random Forest model with the best hyperparameters
    - best_params: The best hyperparameters found by GridSearchCV
    - dev_accuracy: Accuracy of the best model on the development set
    """
    # Split features and target for training and development datasets
    X_train = train_df.drop(target_column, axis=1)
    y_train = train_df[target_column]

    # Define the parameter grid for hyperparameter tuning
    param_grid = {
        'n_estimators': [100, 200, 300],  # Number of trees in the forest
        'max_depth': [10, 20],      # Maximum depth of the tree
        'min_samples_split': [2, 5, 10],  # Minimum number of samples required to split an internal node
        'min_samples_leaf': [2, 4],    # Minimum number of samples required to be at a leaf node
        'bootstrap': [True, False]        # Whether bootstrap samples are used when building trees
    }

    # Initialize the RandomForestClassifier
    rf_model = RandomForestClassifier(random_state=42)

    # Set up the GridSearchCV for hyperparameter tuning
    grid_search = GridSearchCV(estimator=rf_model, param_grid=param_grid, cv=5, scoring='f1_weighted', n_jobs=-1, verbose=2)

    # Fit GridSearchCV
    grid_search.fit(X_train, y_train)

    # Get the best model and parameters from grid search
    best_model = grid_search.best_estimator_

    return best_model

# Example usage:
if __name__ == "__main__":
    # Load your datasets using your data_loader functions
    train_df, encoder = data_loader.load_train_encode_post()

    best_model = random_forest(train_df)

    with open('random_forest_model.pkl', 'wb') as f:
        pickle.dump(best_model, f)