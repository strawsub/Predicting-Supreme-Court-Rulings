import numpy as np
import pickle
from sklearn.metrics import accuracy_score, classification_report
from sklearn.tree import DecisionTreeClassifier

from utils.data_loader import load_train_full, load_dev_full, load_dev_embed

# Load trained classifiers
with open('../stacking_post_models/decision_tree.pkl', 'rb') as f:
    decision_tree_model = pickle.load(f)
with open('knn.pkl', 'rb') as f:
    knn_model = pickle.load(f)
with open('logistic.pkl', 'rb') as f:
    logistic_regression_model = pickle.load(f)
with open('random_forest_model.pkl', 'rb') as f:
    naive_bayes_model = pickle.load(f)

# Load predictions for stacking
logistic_preds = np.load('../archived/custom_ensemble/stacking_predictions/logistic.npy')
knn_preds = np.load('../archived/custom_ensemble/stacking_predictions/knn.npy')
naive_bayes_preds = np.load('../archived/custom_ensemble/stacking_predictions/naive_bayes.npy')
decision_tree_preds = np.load('../archived/custom_ensemble/stacking_predictions/decision_tree.npy')

# Stack predictions
X_train_stack = np.column_stack([
    logistic_preds,
    knn_preds,
    naive_bayes_preds,
    decision_tree_preds
])

# Load labels
y_train_stack = load_train_full()['successful_appeal']
X_dev = load_dev_embed()
y_dev = load_dev_full()['successful_appeal']

# Meta-classifier: decision tree
meta_classifier = DecisionTreeClassifier()
meta_classifier.fit(X_train_stack, y_train_stack)

# Get predictions for dev set from each base classifier
logistic_dev_preds = logistic_regression_model.predict(X_dev)
knn_dev_preds = knn_model.predict(X_dev)
naive_bayes_dev_preds = naive_bayes_model.predict(X_dev)
decision_tree_dev_preds = decision_tree_model.predict(X_dev)

# Stack dev predictions
X_dev_stack = np.column_stack([
    logistic_dev_preds,
    knn_dev_preds,
    naive_bayes_dev_preds,
    decision_tree_dev_preds
])

# Evaluate the meta-classifier
y_dev_pred = meta_classifier.predict(X_dev_stack)
accuracy = accuracy_score(y_dev, y_dev_pred)
print(f'Stacking Model Accuracy: {accuracy:.4f}')

# Print classification report
print("Classification Report:")
print(classification_report(y_dev, y_dev_pred))
