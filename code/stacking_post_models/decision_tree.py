import pickle
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from utils import data_loader, learning_curve


def decision_tree(train_df, target_column='successful_appeal'):
    """
    Perform hyperparameter tuning for Decision Tree using GridSearchCV.

    Parameters:
    - train_df: Training dataset as a pandas DataFrame
    - target_column: The column name of the target variable (default is 'successful_appeal')

    Returns:
    - best_model: The Decision Tree model with the best hyperparameters
    - best_params: The best hyperparameters found by GridSearchCV
    """
    # Split features and target for training dataset
    X_train = train_df.drop(target_column, axis=1)
    y_train = train_df[target_column]

    # Define the parameter grid for hyperparameter tuning
    param_grid = {
        'max_depth': [10, 20, 30],             # Maximum depth of the tree
        'min_samples_split': [2, 5, 10],             # Minimum number of samples required to split an internal node
        'min_samples_leaf': [1, 2, 4],               # Minimum number of samples required to be at a leaf node
        'criterion': ['gini', 'entropy']             # The function to measure the quality of a split
    }

    # Initialize the DecisionTreeClassifier
    dt_model = DecisionTreeClassifier(random_state=42)
    learning_curve.plot_validation_curve(dt_model, train_df, 'max_depth', [1, 5, 10, 20])
    # Set up the GridSearchCV for hyperparameter tuning
    grid_search = GridSearchCV(estimator=dt_model, param_grid=param_grid, cv=5, scoring='f1_weighted', n_jobs=-1, verbose=2)

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
    train_df, encoder = data_loader.load_train_encode_post()
    # Best Parameters: {'criterion': 'entropy', 'max_depth': 10, 'min_samples_leaf': 4, 'min_samples_split': 2}
    best_model = decision_tree(train_df)

    with open('../stacking_pre_models/decision_tree_model.pkl', 'wb') as f:
        pickle.dump(best_model, f)