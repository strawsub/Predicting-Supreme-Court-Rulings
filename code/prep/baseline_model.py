from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, classification_report
from utils import data_loader


def majority_baseline_model(train_df, dev_df, target_column='successful_appeal'):
    """
    Train a majority baseline model and evaluate its accuracy on the development set.

    Parameters:
    - train_df: Training dataset as a pandas DataFrame
    - dev_df: Development (pre_decision) dataset as a pandas DataFrame
    - target_column: The column name of the target variable (default is 'successful_appeal')

    Returns:
    - baseline_accuracy: The accuracy of the majority baseline model on the dev set
    """
    # Split features and target for training and development datasets
    X_train = train_df.drop(target_column, axis=1)
    y_train = train_df[target_column]
    X_dev = dev_df.drop(target_column, axis=1)
    y_dev = dev_df[target_column]

    # Define and train the majority baseline classifier
    majority_baseline = DummyClassifier(strategy='most_frequent')
    majority_baseline.fit(X_train, y_train)

    # Predict on the development set
    y_pred_baseline = majority_baseline.predict(X_dev)

    # Calculate accuracy
    baseline_accuracy = accuracy_score(y_dev, y_pred_baseline)
    print(classification_report(y_dev, y_pred_baseline))
    return baseline_accuracy


if __name__ == "__main__":
    # Load your datasets using your data_loader functions
    train_df_pre = data_loader.load_train_pre()
    dev_df_pre = data_loader.load_dev_pre()
    print("Pre-hearing features, baseline accuracy: ", majority_baseline_model(train_df_pre, dev_df_pre))
