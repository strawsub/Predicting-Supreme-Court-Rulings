import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
import datetime

important_features_post = [
    'majority_ratio',
    'respondent_category_United States',
    'deliberation_length',
    'decision_date_num',
    'argument_date_num',
    'court_hearing_length',
    'utterances_number',
    'petitioner_category_United States',
    'issue_area_Judicial Power',
    'year',
    'successful_appeal'
]

important_features_test = [
    'majority_ratio',
    'respondent_category_United States',
    'deliberation_length',
    'decision_date_num',
    'argument_date_num',
    'court_hearing_length',
    'utterances_number',
    'petitioner_category_United States',
    'issue_area_Judicial Power',
    'year',
]

important_features_pre = [
    'respondent_category_United States',
    'argument_date_num',
    'court_hearing_length',
    'utterances_number',
    'petitioner_category_United States',
    'issue_area_Judicial Power',
    'year',
    'successful_appeal'
]



# Load raw files
def load_train_pre():
    data = pd.read_json('../data/train.jsonl', lines=True)
    pre_hearing_columns = [
        'title', 'petitioner', 'respondent',
        'petitioner_state', 'respondent_state', 'petitioner_category', 'utterances_number',
        'respondent_category', 'issue_area', 'year', 'argument_date', 'successful_appeal', 'court_hearing_length'
    ]
    data = data[pre_hearing_columns]
    return impute_missing_values(data)


def load_dev_pre():
    data = pd.read_json('../data/dev.jsonl', lines=True)
    pre_hearing_columns = [
        'title', 'petitioner', 'respondent',
        'petitioner_state', 'respondent_state', 'petitioner_category', 'utterances_number',  'court_hearing_length',
        'respondent_category', 'issue_area', 'year', 'argument_date', 'successful_appeal'
    ]
    data = data[pre_hearing_columns]
    return impute_missing_values(data)


def load_test_pre():
    data = pd.read_json('../data/test.jsonl', lines=True)
    pre_hearing_columns = [
        'title', 'petitioner', 'respondent',
        'petitioner_state', 'respondent_state', 'petitioner_category', 'utterances_number',  'court_hearing_length',
        'respondent_category', 'issue_area', 'year', 'argument_date'
    ]
    data = data[pre_hearing_columns]
    return impute_missing_values(data)

def load_train_post():
    data = pd.read_json('../data/train.jsonl', lines=True)
    post_hearing_columns = [
        'title', 'petitioner', 'respondent',
        'petitioner_state', 'respondent_state', 'petitioner_category', 'utterances_number', 'court_hearing_length',
        'respondent_category', 'issue_area', 'year', 'argument_date', 'decision_date', 'majority_ratio', 'successful_appeal'
    ]
    data = data[post_hearing_columns]
    return impute_missing_values(data)

def load_dev_post():
    data = pd.read_json('../data/dev.jsonl', lines=True)
    post_hearing_columns = [
        'title', 'petitioner', 'respondent',
        'petitioner_state', 'respondent_state', 'petitioner_category', 'utterances_number',  'court_hearing_length',
        'respondent_category', 'issue_area', 'year', 'argument_date', 'successful_appeal', 'decision_date', 'majority_ratio'
    ]
    data = data[post_hearing_columns]
    return impute_missing_values(data)

def load_test_post():
    data = pd.read_json('../data/test.jsonl', lines=True)
    post_hearing_columns = [
        'title', 'petitioner', 'respondent',
        'petitioner_state', 'respondent_state', 'petitioner_category', 'utterances_number',
        'respondent_category', 'issue_area', 'year', 'argument_date', 'decision_date', 'majority_ratio'
    ]
    data = data[post_hearing_columns]
    return impute_missing_values(data)


def load_train_full():
    return pd.read_json('../data/train.jsonl', lines=True)


def load_dev_full():
    return pd.read_json('../data/dev.jsonl', lines=True)


def load_test_full():
    data = pd.read_json('../data/test.jsonl', lines=True)
    return impute_missing_values(data)


# Function to impute missing values with 'UNKNOWN'
def impute_missing_values(df):
    df.fillna("UNKNOWN", inplace=True)
    return df


# Function to count the number of observations (rows) in the dataset
def count_observations(df, name):
    print(f"Number of observations in {name} dataset: {df.shape[0]}")


# Function to check for missing values
def check_missing_values(df, name):
    print(f"Missing values in {name} dataset:")
    print(df.isnull().sum())
    print("\n")


# Function to check for 'UNKNOWN' values
def check_unknown_values(df, name):
    print(f"Checking for 'UNKNOWN' values in {name} dataset:")
    unknown_counts = df.apply(lambda col: col.str.contains('UNKNOWN', na=False).sum() if col.dtype == 'object' else 0)
    print(unknown_counts)
    print("\n")

# Load embeds
def load_embed():
    data = {
        'train': np.load('../data/sembed/train.npy'),
        'dev': np.load('../data/sembed/dev.npy'),
        'pre_decision': np.load('../data/sembed/test.npy')
    }
    return data


def load_train_embed():
    return np.load('../data/sembed/train.npy')


def load_dev_embed():
    return np.load('../data/sembed/dev.npy')


def load_test_embed():
    return np.load('../data/sembed/test.npy')


def feature_engineer(data, encoder=None):
    # Columns to drop if they exist in the data
    columns_to_drop = ['title', 'petitioner', 'respondent', 'court_hearing', 'case_id']
    data = data.drop(columns=[col for col in columns_to_drop if col in data.columns])

    # One-hot encoding for these columns if they exist
    columns_to_encode = ['petitioner_state', 'respondent_state', 'petitioner_category',
                         'respondent_category', 'issue_area', 'chief_justice']

    # Check which columns are present in the data
    columns_to_encode = [col for col in columns_to_encode if col in data.columns]

    if encoder is None and columns_to_encode:
        # Create a new encoder and fit it on the available columns
        encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        encoded_features = pd.DataFrame(encoder.fit_transform(data[columns_to_encode]),
                                        columns=encoder.get_feature_names_out(columns_to_encode))
    elif columns_to_encode:
        # Use the already fitted encoder to transform the available columns
        encoded_features = pd.DataFrame(encoder.transform(data[columns_to_encode]),
                                        columns=encoder.get_feature_names_out(columns_to_encode))
    else:
        # No columns to encode, return the data as is
        encoded_features = pd.DataFrame()

    # Drop the original columns that were one-hot encoded (if any)
    data = data.drop(columns=columns_to_encode, errors='ignore')

    # Convert argument_date to numerical format (if it exists)
    if 'argument_date' in data.columns:

        data['argument_date'] = pd.to_datetime(data['argument_date'], errors='coerce')
        reference_date = datetime.datetime(1900, 1, 1)
        data['argument_date_num'] = (data['argument_date'] - reference_date).dt.days
        data = data.drop(columns=['argument_date'])

    # Convert decision_date to numerical format (if it exists)
    if 'decision_date' in data.columns:
        data['decision_date'] = pd.to_datetime(data['decision_date'], errors='coerce')
        data['decision_date_num'] = (data['decision_date'] - reference_date).dt.days
        data = data.drop(columns=['decision_date'])

    if 'argument_date_num' in data.columns and 'decision_date_num' in data.columns:
        data['deliberation_length'] = data['decision_date_num'] - data['argument_date_num']

    # Unpack the justices' information by counting gender and political direction (if the column exists)
    if 'justices' in data.columns:
        def count_justices(justices):
            num_males = sum(1 for j in justices if j['gender'] == 'male')
            num_females = sum(1 for j in justices if j['gender'] == 'female')
            num_liberal = sum(1 for j in justices if j['political_direction'] == 'Liberal')
            num_conservative = sum(1 for j in justices if j['political_direction'] == 'Conservative')
            return pd.Series([num_males, num_females, num_liberal, num_conservative])

        data[['num_males', 'num_females', 'num_liberal', 'num_conservative']] = data['justices'].apply(count_justices)
        data = data.drop(columns=['justices'])


    # Concatenate the one-hot encoded features with the original data
    if not encoded_features.empty:
        final_data = pd.concat([data, encoded_features], axis=1)
    else:
        final_data = data

    return final_data, encoder

# Combine raw & embed, with encoding
def load_train_encode_all():
    train = load_train_full()
    train, encoder = feature_engineer(train)
    embeds = pd.DataFrame(load_train_embed())
    df = pd.concat([train, embeds], axis=1)
    df.columns = df.columns.astype(str)
    print("Train (full) dataframe size ", df.shape)
    return df, encoder  # Return the encoder for further use

def load_train_encode_full():
    train = load_train_full()
    train, encoder = feature_engineer(train)
    train = train[important_features_post]
    embeds = pd.DataFrame(load_train_embed())
    df = pd.concat([train, embeds], axis=1)
    df.columns = df.columns.astype(str)
    print("Train (full) dataframe size ", df.shape)
    return df, encoder  # Return the encoder for further use


def load_dev_encode_full(encoder):
    dev = load_dev_full()[important_features_post]
    dev = dev[important_features_post]
    dev, _ = feature_engineer(dev, encoder)  # Use the same encoder
    embeds = pd.DataFrame(load_dev_embed())
    df = pd.concat([dev, embeds], axis=1)
    df.columns = df.columns.astype(str)
    print("Dev (full) dataframe size ", df.shape)
    return df


def load_test_encode_full(encoder):
    test = load_test_full()
    test, _ = feature_engineer(test, encoder)  # Use the same encoder
    test = test[important_features_test]
    embeds = pd.DataFrame(load_test_embed())
    df = pd.concat([test, embeds], axis=1)
    df.columns = df.columns.astype(str)
    print("Test (full) dataframe size ", df.shape)
    return df


def load_train_encode_pre():
    train = load_train_pre()
    train, encoder = feature_engineer(train)

    train = train[important_features_pre]
    embeds = pd.DataFrame(load_train_embed())
    df = pd.concat([train, embeds], axis=1)
    df.columns = df.columns.astype(str)
    print("Train (pre-hearing) dataframe size ", df.shape)
    return df, encoder  # Return the encoder for further use


def load_dev_encode_pre(encoder):
    dev = load_dev_pre()
    dev, _ = feature_engineer(dev, encoder)  # Use the same encoder
    dev = dev[important_features_pre]
    embeds = pd.DataFrame(load_dev_embed())
    df = pd.concat([dev, embeds], axis=1)
    df.columns = df.columns.astype(str)
    print("Dev (pre-hearing) dataframe size ", df.shape)
    return df


def load_test_encode_pre(encoder):
    test = load_test_full()
    test, _ = feature_engineer(test, encoder)  # Use the same encoder
    test = test[important_features_pre]
    embeds = pd.DataFrame(load_test_embed())
    df = pd.concat([test, embeds], axis=1)
    df.columns = df.columns.astype(str)
    print("Test (pre-hearing) dataframe size ", df.shape)
    return df


def load_train_encode_post():
    train = load_train_post()
    train, encoder = feature_engineer(train)
    train = train[important_features_post]
    embeds = pd.DataFrame(load_train_embed())
    df = pd.concat([train, embeds], axis=1)
    df.columns = df.columns.astype(str)
    print("Train (post-hearing) dataframe size ", df.shape)
    return df, encoder  # Return the encoder for further use


def load_dev_encode_post(encoder):
    dev = load_dev_post()
    dev, _ = feature_engineer(dev, encoder)  # Use the same encoder
    dev = dev[important_features_post]
    embeds = pd.DataFrame(load_dev_embed())
    df = pd.concat([dev, embeds], axis=1)
    df.columns = df.columns.astype(str)
    print("Dev (post-hearing) dataframe size ", df.shape)
    return df


def load_test_encode_post(encoder):
    test = load_test_post()
    test, _ = feature_engineer(test, encoder)  # Use the same encoder
    test = test[important_features_post]
    embeds = pd.DataFrame(load_test_embed())
    df = pd.concat([test, embeds], axis=1)
    df.columns = df.columns.astype(str)
    print("Test (post-hearing) dataframe size ", df.shape)
    return df


if __name__ == "__main__":
    train_df_post, encoder_post = load_train_encode_post()
    dev_df_post = load_dev_encode_post(encoder_post)