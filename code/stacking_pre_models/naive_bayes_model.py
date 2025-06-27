import pickle
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import GridSearchCV
from utils import data_loader

def naive_bayes(train_df, target_column='successful_appeal'):
    """
    Perform hyperparameter tuning for Naive Bayes using GridSearchCV.

    Parameters:
    - train_df: Training dataset as a pandas DataFrame
    - target_column: The column name of the target variable (default is 'successful_appeal')

    Returns:
    - best_model: The Naive Bayes model with the best hyperparameters
    - best_params: The best hyperparameters found by GridSearchCV
    """
    # Split features and target for training dataset
    X_train = train_df.drop(target_column, axis=1)
    y_train = train_df[target_column]

    # Define the parameter grid for hyperparameter tuning
    param_grid = {
        'var_smoothing': [1e-9, 1e-8, 1e-7, 1e-6]  # Smoothing parameter to avoid zero probabilities
    }

    # Initialize the GaussianNB model
    nb_model = GaussianNB()

    # Set up the GridSearchCV for hyperparameter tuning
    grid_search = GridSearchCV(estimator=nb_model, param_grid=param_grid, cv=5, scoring='f1_weighted', n_jobs=-1, verbose=2)

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

    best_model = naive_bayes(train_df) # smoothing 1e-8
    with open('naive_bayes_model.pkl', 'wb') as f:
        pickle.dump(best_model, f)