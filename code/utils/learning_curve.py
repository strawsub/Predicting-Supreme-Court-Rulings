import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import learning_curve, validation_curve
from sklearn.preprocessing import StandardScaler


def plot_learning_curve_holdout(model, train_df, test_df):
    X_train = train_df.drop('successful_appeal', axis=1)
    y_train = train_df['successful_appeal']

    X_test = test_df.drop('successful_appeal', axis=1)
    y_test = test_df['successful_appeal']

    # Get learning curve data
    train_sizes, train_scores, test_scores = learning_curve(
        model, X_train, y_train, cv=5, n_jobs=-1,
        train_sizes=np.linspace(0.1, 1.0, 10), scoring='accuracy'
    )

    # Calculate mean and std for training scores
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)

    # Calculate accuracy for test set
    test_scores_mean = []
    test_scores_std = []

    for train_size in train_sizes:
        # Fit the model on a subset of the training data
        model.fit(X_train[:int(train_size * len(X_train))], y_train[:int(train_size * len(y_train))])

        # Evaluate the model on the test set
        test_score = model.score(X_test, y_test)
        test_scores_mean.append(test_score)
        test_scores_std.append(0)  # Standard deviation is not applicable for a single test score

    # Convert lists to numpy arrays for easy plotting
    test_scores_mean = np.array(test_scores_mean)
    test_scores_std = np.array(test_scores_std)

    # Plotting the learning curve
    plt.figure(figsize=(10, 6))
    plt.plot(train_sizes, train_scores_mean, label='Training Score', color='blue')
    plt.plot(train_sizes, test_scores_mean, label='Test Score', color='green')

    # Plot the std deviation as shaded area
    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, color='blue', alpha=0.1)
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, color='green', alpha=0.1)

    plt.title('Learning Curve (Hold-Out)')
    plt.xlabel('Training Size')
    plt.ylabel('Accuracy')
    plt.legend(loc='best')
    plt.grid()
    plt.show()

def plot_learning_curve_cv(model, data, title="Learning Curve"):
    X = data.drop('successful_appeal', axis=1)
    y = data['successful_appeal']
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    train_sizes, train_scores, test_scores = learning_curve(
        model, X, y, cv=5, n_jobs=-1,
        train_sizes=np.linspace(0.1, 1.0, 10), scoring='f1'
    )

    # Calculate mean and std for training and test scores
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    # Plotting the learning curve
    plt.figure(figsize=(10, 6))
    plt.plot(train_sizes, train_scores_mean, label='Training Score', color='blue')
    plt.plot(train_sizes, test_scores_mean, label='Validation Score', color='green')

    # Plot the std deviation as shaded area
    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, color='blue', alpha=0.1)
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, color='green', alpha=0.1)

    plt.title(title)
    plt.xlabel('Training Size')
    plt.ylabel('F1 Score')
    plt.legend(loc='best')
    plt.grid()
    plt.show()


def plot_validation_curve(model, data, param_name, param_range, cv=5):
    # Separate features and target variable
    X = data.drop('successful_appeal', axis=1)
    y = data['successful_appeal']

    # Standardize features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Calculate the training and validation scores
    train_scores, val_scores = validation_curve(
        model, X, y, param_name=param_name, param_range=param_range, scoring="f1", cv=cv, n_jobs=-1
    )

    # Compute the mean and standard deviation for both sets of scores
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    val_mean = np.mean(val_scores, axis=1)
    val_std = np.std(val_scores, axis=1)

    # Plot validation curve
    plt.figure(figsize=(8, 6))
    plt.plot(param_range, train_mean, label="Training score", color="r")
    plt.fill_between(param_range, train_mean - train_std, train_mean + train_std, color="r", alpha=0.2)

    plt.plot(param_range, val_mean, label="Validation score", color="g")
    plt.fill_between(param_range, val_mean - val_std, val_mean + val_std, color="g", alpha=0.2)

    plt.title(f"Validation Curve for {model.__class__.__name__}")
    plt.xlabel(param_name)
    plt.ylabel("F1 Score")
    plt.xscale('log') if param_name in ['min_samples_split', 'min_samples_leaf'] else plt.xscale('linear')
    plt.legend(loc="best")
    plt.grid(True)
    plt.show()
