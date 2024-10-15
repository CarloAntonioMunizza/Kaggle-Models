# Importing necessary libraries
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

# Load data
def SetPaths(): # takes no parameters, returns 5 values, root_path, train_identity_path, train_transaction_path, test_identity_path, test_transaction_path
    root_path = 'C:/Users/carlo/.vscode/Repos/Kaggle Models/Kaggle Data/IEEE-CIS Fraud Detection Data/'
    train_identity_path = root_path + 'train_identity.csv'
    train_transaction_path = root_path + 'train_transaction.csv'
    test_identity_path = root_path + 'test_identity.csv'
    test_transaction_path = root_path + 'test_transaction.csv'
    return root_path, train_identity_path, train_transaction_path, test_identity_path, test_transaction_path

# Fill empty columns with NaN
def fill_empty_columns(df):
    return df.fillna(np.nan)


# Read CSV files into DataFrames and clean them
def CreateDataFrames(train_identity_path, train_transaction_path, test_identity_path, test_transaction_path): # takes 4 paths to csv and returns 2 concated dataframes
    train_identity_data = pd.read_csv(train_identity_path)
    train_transaction_data = pd.read_csv(train_transaction_path)
    test_identity_data = pd.read_csv(test_identity_path)
    test_transaction_data = pd.read_csv(test_transaction_path)

    # Check if DataFrames are empty
    for df_name, df in zip(['Train Identity', 'Train Transaction', 'Test Identity', 'Test Transaction'],
                           [train_identity_data, train_transaction_data, test_identity_data, test_transaction_data]):
        if df.empty:
            raise ValueError(f"{df_name} is empty. Please check the file path or file content.")

    # Rename columns to replace underscores with hyphens
    train_identity_data.columns = train_identity_data.columns.str.replace('_', '-', regex=False)
    train_transaction_data.columns = train_transaction_data.columns.str.replace('_', '-', regex=False)
    test_identity_data.columns = test_identity_data.columns.str.replace('_', '-', regex=False)
    test_transaction_data.columns = test_transaction_data.columns.str.replace('_', '-', regex=False)

    # Filter identity and transaction data
    train_filtered_identity_data = train_identity_data[train_identity_data['TransactionID'].isin(train_transaction_data['TransactionID'])]
    test_filtered_identity_data = test_identity_data[test_identity_data['TransactionID'].isin(test_transaction_data['TransactionID'])]

    train_filtered_transaction_data = train_transaction_data[train_transaction_data['TransactionID'].isin(train_filtered_identity_data['TransactionID'])]
    test_filtered_transaction_data = test_transaction_data[test_transaction_data['TransactionID'].isin(test_filtered_identity_data['TransactionID'])]

    # Combine identity and transaction data for train and test
    train_combined_data = pd.merge(train_filtered_transaction_data, train_filtered_identity_data, on='TransactionID', how='outer')
    test_combined_data = pd.merge(test_filtered_transaction_data, test_filtered_identity_data, on='TransactionID', how='outer')

    # Convert all data to numeric (this will handle NaNs)
    train_combined_data = train_combined_data.apply(pd.to_numeric, errors='coerce')
    test_combined_data = test_combined_data.apply(pd.to_numeric, errors='coerce')

    # Fill columns with NaN or a custom placeholder (-999) where applicable
    train_combined_data = fill_empty_columns(train_combined_data)
    test_combined_data = fill_empty_columns(test_combined_data)

    # Handle infinite values and large values
    train_combined_data.replace([np.inf, -np.inf], np.nan, inplace=True)
    test_combined_data.replace([np.inf, -np.inf], np.nan, inplace=True)
    max_value_threshold = 1e10
    train_combined_data[train_combined_data > max_value_threshold] = max_value_threshold
    test_combined_data[test_combined_data > max_value_threshold] = max_value_threshold

    # Check for columns with all missing values
    missing_columns = train_combined_data.columns[train_combined_data.isnull().all()]
    if len(missing_columns) > 0:
        #print("Dropping columns with all missing values:", missing_columns)
        train_combined_data.drop(columns=missing_columns, inplace=True)
        test_combined_data.drop(columns=missing_columns, inplace=True)

    # Separate labels before imputation
    Y_train_combined_data = train_combined_data['isFraud']
    X_train_combined_data = train_combined_data.drop('isFraud', axis=1)

    # Apply the imputer
    imputer = SimpleImputer(strategy='mean')
    X_train_combined_data_imputed = imputer.fit_transform(X_train_combined_data)
    X_test_combined_data_imputed = imputer.transform(test_combined_data)

    # Recreate DataFrames
    test_combined_data_imputed = pd.DataFrame(X_test_combined_data_imputed, columns=test_combined_data.columns)
    train_combined_data_imputed = pd.DataFrame(X_train_combined_data_imputed, columns=X_train_combined_data.columns)
    train_combined_data_imputed['isFraud'] = Y_train_combined_data.values

    return train_combined_data_imputed, test_combined_data_imputed

# Set up split and final dataframes
def TrainValTestSplit(train_combined_data, test_combined_data): # takes 2 dataframes, one for training data and other for test data and returns 5 data frames
    X_train_combined_data = train_combined_data.drop('isFraud', axis=1)
    Y_train_combined_data = train_combined_data['isFraud']

    # No need for another imputer, as it's already handled in CreateDataFrames
    non_missing_columns = X_train_combined_data.columns

    # Train-test split
    X_train, X_val, Y_train, Y_val = train_test_split(
        X_train_combined_data, Y_train_combined_data, test_size=0.2, stratify=Y_train_combined_data, random_state=1
    )

    # Standardize features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(test_combined_data[non_missing_columns])

    # Convert back to DataFrames with original feature names
    X_train = pd.DataFrame(X_train, columns=non_missing_columns)
    X_val = pd.DataFrame(X_val, columns=non_missing_columns)
    X_test = pd.DataFrame(X_test, columns=non_missing_columns)

    return X_train, X_val, Y_train, Y_val, X_test

# initilize logistical regression model
def LogRegInit(): # takes no params and returns model
    log_reg_model = LogisticRegression(
        penalty='l2',
        C=1.0,
        solver='lbfgs',
        max_iter=5000,
        random_state=42,
        class_weight='balanced'
    )
    return log_reg_model

# Generate scoring metrics
def GenerateMetrics(TrueLabel, PredLabel): # takes 2 params, being the true labels and the generated labels, prints scores to screen returns nothing
    f1 = f1_score(TrueLabel, PredLabel)
    accuracy = accuracy_score(TrueLabel, PredLabel)

    print('F1 Score:', f1)
    print('Accuracy:', accuracy)

# Generate Submission CSV
def GenerateSubmission(root_path): # takes root path and creates submission csv based off the root
    # Assuming test_combined_data_imputed is your test set
    Y_submission = log_reg_model.predict(X_test)

    # Creating a DataFrame for submission
    submission = pd.DataFrame({
        'TransactionID': test_combined_data['TransactionID'],  # Ensure this column exists
        'isFraud': Y_submission
    })

    # Save the predictions into a CSV file for submission
    submission.to_csv(root_path + 'submission.csv', index=False)


# Create basic paths
root_path, train_identity_path, train_transaction_path, test_identity_path, test_transaction_path = SetPaths()

# Create concatenated dataframes
train_combined_data, test_combined_data = CreateDataFrames(train_identity_path, train_transaction_path, test_identity_path, test_transaction_path)

# Create train val test splits
X_train, X_val, Y_train, Y_val, X_test = TrainValTestSplit(train_combined_data, test_combined_data)

# Initilize model
log_reg_model = LogRegInit()

# Fit the model
log_reg_model.fit(X_train, Y_train)

# Predict
Y_pred = log_reg_model.predict(X_train)

# Evaluate
GenerateMetrics(Y_train, Y_pred)

# Predict
Y_pred = log_reg_model.predict(X_val)

# Evaluate
GenerateMetrics(Y_val, Y_pred)

# Create Submission
GenerateSubmission(root_path)