# Importing necessary libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

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

# Apply PCA
pca = PCA(n_components=10)  # Start with 10 components
X_train_pca = pca.fit_transform(X_train_combined_data)

# Get the explained variance ratio of each principal component
explained_variance = pca.explained_variance_ratio_

# Plot the explained variance to observe how much information each component holds
plt.figure(figsize=(10,6))
plt.bar(range(1, len(explained_variance)+1), explained_variance, alpha=0.7)
plt.ylabel('Explained Variance Ratio')
plt.xlabel('Principal Component')
plt.title('PCA Explained Variance')
plt.show()

# Analyze the most important features contributing to the top components
components = pd.DataFrame(pca.components_, columns=non_missing_columns)
top_features = components.abs().sum().sort_values(ascending=False).head(20)
print(f"Top 10 important features based on PCA:\n{top_features}")