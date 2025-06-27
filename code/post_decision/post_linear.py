
import pandas as pd
import statsmodels.api as sm
import seaborn as sns
import statsmodels.formula.api as smf
from matplotlib import pyplot as plt
from statsmodels.stats.anova import anova_lm
import numpy as np
from utils import data_loader

df = data_loader.load_train_encode_full()
print(df[0]['deliberation_length'].dtype)

def create_log_odds_plot(df, predictor, outcome, n_bins=10):
    df = df[0]
    # Create bins for the predictor
    df['bins'] = pd.cut(df[predictor], bins=n_bins)

    # Calculate the mean of the predictor and the proportion of positive outcomes for each bin
    grouped = df.groupby('bins').agg({predictor: 'mean', outcome: 'mean'}).reset_index()

    # Calculate log-odds (adding a small value to avoid log(0))
    grouped['log_odds'] = np.log(grouped[outcome].clip(lower=0.001, upper=0.999) /
                                (1 - grouped[outcome].clip(lower=0.001, upper=0.999)))
    grouped = grouped.dropna()
    # Create the scatter plot
    plt.figure(figsize=(10, 6))
    plt.scatter(grouped[predictor], grouped['log_odds'])
    plt.xlabel(predictor)
    plt.ylabel('Log-odds of successful_appeal')
    plt.title(f'Log-odds of successful_appeal vs. {predictor}')

    # Fit a line to check for linearity
    coeffs = np.polyfit(grouped[predictor], grouped['log_odds'], deg=1)
    plt.plot(grouped[predictor], np.polyval(coeffs, grouped[predictor]), 'r--')

    plt.show()

# Usage
create_log_odds_plot(df, 'deliberation_length', 'successful_appeal')