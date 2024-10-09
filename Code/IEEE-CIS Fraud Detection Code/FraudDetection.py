# Importing necessary libraries
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from sklearn.impute import SimpleImputer

# Load data
root_path = 'C:/Users/carlo/.vscode/Repos/Kaggle Models/Kaggle Data/IEEE-CIS Fraud Detection Data/'
train_identity_path = root_path + 'train_identity.csv'
train_transaction_path = root_path + 'train_transaction.csv'
test_identity_path = root_path + 'test_identity.csv'
test_transaction_path = root_path + 'test_transaction.csv'

# Read CSV files into DataFrames
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

# Combine identity and transaction data
train_combined_data = pd.concat([train_filtered_transaction_data, train_filtered_identity_data], axis=1)
test_combined_data = pd.concat([test_filtered_transaction_data, test_filtered_identity_data], axis=1)

# Ensure all columns are numeric and check if 'isFraud' exists
if 'isFraud' not in train_combined_data.columns:
    raise ValueError("'isFraud' column not found in the training data.")

# Remove rows with NaN in 'isFraud'
train_combined_data = train_combined_data.dropna(subset=['isFraud'])

# Combine identity and transaction data for train and test
train_combined_data = pd.merge(train_transaction_data, train_identity_data, on='TransactionID', how='outer')
test_combined_data = pd.merge(test_transaction_data, test_identity_data, on='TransactionID', how='outer')

# Ensure all columns are numeric and check if 'isFraud' exists in train_combined_data
if 'isFraud' not in train_combined_data.columns:
    raise ValueError("'isFraud' column not found in the training data.")

# Remove rows with NaN in 'isFraud'
train_combined_data = train_combined_data.dropna(subset=['isFraud'])

X_train_combined_data = train_combined_data.drop('isFraud', axis=1)
Y_train_combined_data = train_combined_data['isFraud']

# Convert all data to numeric (this will handle NaNs)
X_train_combined_data = X_train_combined_data.apply(pd.to_numeric, errors='coerce')
test_combined_data = test_combined_data.apply(pd.to_numeric, errors='coerce')

# Handle infinite values and large values
X_train_combined_data.replace([np.inf, -np.inf], np.nan, inplace=True)
test_combined_data.replace([np.inf, -np.inf], np.nan, inplace=True)
max_value_threshold = 1e10
X_train_combined_data[X_train_combined_data > max_value_threshold] = max_value_threshold
test_combined_data[test_combined_data > max_value_threshold] = max_value_threshold

# Handle missing values in X_train_combined_data using SimpleImputer
imputer = SimpleImputer(strategy='mean')

# Check for columns with no non-missing values
non_missing_columns = X_train_combined_data.columns[X_train_combined_data.notnull().any()]
X_train_combined_data = imputer.fit_transform(X_train_combined_data[non_missing_columns])

# Ensure test_combined_data has the same columns as X_train_combined_data
test_combined_data_imputed = pd.DataFrame(
    imputer.transform(test_combined_data[non_missing_columns]),
    columns=non_missing_columns
)

# Train-test split
X_train, X_val, Y_train, Y_val = train_test_split(
    X_train_combined_data, Y_train_combined_data, test_size=0.2, stratify=Y_train_combined_data, random_state=1
)

# Initialize and train the logistic regression model
log_reg_model = LogisticRegression(
    penalty='l2',
    C=1.0,
    solver='lbfgs',
    max_iter=20000,
    random_state=42,
    class_weight='balanced'
)
# Fit the model
log_reg_model.fit(X_train, Y_train)

# Print the number of iterations
print("Number of iterations:", log_reg_model.n_iter_)

# Predict and evaluate
Y_pred = log_reg_model.predict(X_train)
f1 = f1_score(Y_train, Y_pred)
accuracy = accuracy_score(Y_train, Y_pred)

print('Train F1 Score:', f1)
print('Train Accuracy:', accuracy)

# Predict and evaluate
Y_pred = log_reg_model.predict(X_val)
f1 = f1_score(Y_val, Y_pred)
accuracy = accuracy_score(Y_val, Y_pred)

print('Val F1 Score:', f1)
print('Val Accuracy:', accuracy)

# Assuming test_combined_data_imputed is your test set (without labels)
Y_submission = log_reg_model.predict(test_combined_data_imputed)

# Creating a DataFrame for submission
# Assuming 'TransactionID' is part of your test_combined_data
submission = pd.DataFrame({
    'TransactionID': test_combined_data['TransactionID'],  # Ensure this column exists
    'isFraud': Y_submission
})

# Save the predictions into a CSV file for submission
submission.to_csv('C:/Users/carlo/.vscode/Repos/Kaggle Models/Kaggle Data/IEEE-CIS Fraud Detection Data/submission.csv', index=False)
