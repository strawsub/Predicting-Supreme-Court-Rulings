import pandas as pd
from sklearn.linear_model import LogisticRegression
import numpy as np
from sklearn.preprocessing import StandardScaler
from utils import data_loader, learning_curve
from pre_decision import logistic_model

def identify_near_boundary_instances(train_data, threshold=0.1):
    X_train, y_train = train_data.drop('successful_appeal', axis=1), train_data['successful_appeal']
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    # Fit the classifier
    clf = LogisticRegression(C=0.1, max_iter=300, solver='lbfgs')
    clf.fit(X_train_scaled, y_train)

    # Get predicted probabilities using scaled features
    probabilities = clf.predict_proba(X_train_scaled)[:, 1]  # Use scaled features for predictions

    # Identify instances near the decision boundary
    near_boundary_indices = np.where((probabilities > (0.5 - threshold)) & (probabilities < (0.5 + threshold)))[0]

    print(f"Identified {len(near_boundary_indices)} instances near the decision boundary.")

    # Exclude the near boundary instances from the training data
    filtered_train_data = train_data.drop(train_data.index[near_boundary_indices])
    near_boundary_df = pd.DataFrame(near_boundary_indices, columns=['near_boundary_indices'])
    near_boundary_df.to_csv('near_boundary_indices.csv', index=False)
    print(f"Near boundary indices exported to near_boundary_indices.csv")

    print(f"Filtered train data: {filtered_train_data.shape}")

    return filtered_train_data

if __name__ == "__main__":
    train_df, encoder = data_loader.load_train_encode_pre()
    dev_df = data_loader.load_dev_encode_pre(encoder)

    # filtered_train_df_1 = identify_near_boundary_instances(train_df)
    # filtered_train_df_2 = identify_near_boundary_instances(filtered_train_df_1)
    filtered_train_df = identify_near_boundary_instances(train_df)
    model = LogisticRegression(max_iter=300, solver='lbfgs')
    learning_curve.plot_learning_curve_cv(model, filtered_train_df)

    logistic_model.best_logistic_classifier(filtered_train_df, dev_df)

