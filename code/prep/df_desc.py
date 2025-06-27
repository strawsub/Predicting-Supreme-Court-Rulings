from matplotlib import pyplot as plt
import seaborn as sns

from utils import data_loader

train_df = data_loader.load_train_full()

dev_df = data_loader.load_dev_full()

# Basic overview
print(train_df.info())
print(train_df.describe())

# # Numerical feature
# train_df['utterances_number'].hist(bins=30, figsize=(15, 10))
# plt.tight_layout()
# plt.show()
#
# # Categorical feature
# categorical_columns = ['petitioner_state', 'respondent_state', 'petitioner_category', 'respondent_category', 'issue_area', 'chief_justice']
# for col in categorical_columns:
#     print(train_df[col].value_counts())

# Date-time features

# Class imbalance check

sns.countplot(x='successful_appeal', data=dev_df)
plt.title('Class Distribution')
plt.xlabel('Successful Appeal')
plt.ylabel('Count')
plt.show()

# Correlation analysis