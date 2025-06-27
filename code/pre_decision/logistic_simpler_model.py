from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from utils import data_loader
from sklearn.model_selection import train_test_split
from utils import learning_curve

# Load and encode training data
train_df_pre, encoder_pre = data_loader.load_train_encode_pre()

# Separate features and labels
X = train_df_pre.drop('successful_appeal', axis=1)
y = train_df_pre['successful_appeal']

# Split data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and fit the model
clf = LogisticRegression(C=1, max_iter=2000)
clf.fit(X_train, y_train)

# Make predictions on the validation set
y_pred = clf.predict(X_val)

# Generate classification report
report = classification_report(y_val, y_pred)

# Print the classification report
print(report)

# Optionally, plot learning curve
learning_curve.plot_learning_curve_cv(clf, train_df_pre)