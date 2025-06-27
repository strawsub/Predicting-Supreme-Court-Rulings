import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import make_scorer, f1_score, precision_score
from utils import data_loader

def precision(X, y, max_depth):
    model = RandomForestClassifier(max_depth=max_depth)
    precision_scorer = make_scorer(precision_score)

    # Perform cross-validation
    precisions = cross_val_score(model, X, y, cv=5, scoring=precision_scorer)
    mean_precision = precisions.mean()

    return mean_precision

def f1(X, y, max_depth):
    model = RandomForestClassifier(max_depth=max_depth)
    f1_scorer = make_scorer(f1_score)

    # Perform cross-validation
    f1_scores = cross_val_score(model, X, y, cv=5, scoring=f1_scorer)
    mean_f1 = f1_scores.mean()

    return mean_f1

if __name__ == "__main__":
    full_df, _ = data_loader.load_train_encode_all()
    no_state_df = full_df.drop(columns=[col for col in full_df.columns if 'state' in col])
    no_category_df = full_df.drop(columns=[col for col in full_df.columns if 'category' in col])
    no_hearing_df, _ = data_loader.feature_engineer(data_loader.load_train_pre())

    datasets = [full_df, no_state_df, no_category_df, no_hearing_df]
    max_depths = [10, 20, 30]

    results = pd.DataFrame(columns=['Dataset', 'Max Depth', 'Precision', 'F1 Score'])
    feature_importances = pd.DataFrame()

    # Precision and F1 Score
    for i, data in enumerate(datasets):
        X = data.drop('successful_appeal', axis=1)
        y = data['successful_appeal']

        for max_depth in max_depths:
            model = RandomForestClassifier(max_depth=max_depth)
            model.fit(X, y)  # Fit the model to get feature importances

            mean_precision = precision(X, y, max_depth)
            mean_f1 = f1(X, y, max_depth)
            new_result = pd.DataFrame({
                'Dataset': [f'Dataset {i + 1}'],
                'Max Depth': [max_depth],
                'Precision': [mean_precision],
                'F1 Score': [mean_f1]
            })
            results = pd.concat([results, new_result], ignore_index=True)

            # Store feature importances
            importance_df = pd.DataFrame({
                'Feature': X.columns,
                'Importance': model.feature_importances_
            })
            importance_df['Dataset'] = f'Dataset {i + 1}'
            importance_df['Max Depth'] = max_depth
            feature_importances = pd.concat([feature_importances, importance_df], ignore_index=True)

    # Export feature importances to CSV
    feature_importances.to_csv('feature_importances.csv', index=False)