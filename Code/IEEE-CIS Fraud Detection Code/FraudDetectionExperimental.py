# Importing necessary libraries
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import optuna  # Import Optuna

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
train_combined_data = pd.merge(train_filtered_transaction_data, train_filtered_identity_data, on='TransactionID', how='inner')
test_combined_data = pd.merge(test_filtered_transaction_data, test_filtered_identity_data, on='TransactionID', how='inner')

# Ensure all columns are numeric and check if 'isFraud' exists
if 'isFraud' not in train_combined_data.columns:
    raise ValueError("'isFraud' column not found in the training data.")

# Remove rows with NaN in 'isFraud'
train_combined_data = train_combined_data.dropna(subset=['isFraud'])

# Extract the relevant features and labels
X_train_combined_data = train_combined_data.drop(columns=['isFraud', 'TransactionID'])  # Exclude the target and ID
Y_train_combined_data = train_combined_data['isFraud']

# Convert all data to numeric (this will handle NaNs)
X_train_combined_data = X_train_combined_data.apply(pd.to_numeric, errors='coerce')
test_combined_data = test_combined_data.drop(columns=['TransactionID']).apply(pd.to_numeric, errors='coerce')

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
X_train_combined_data_imputed = imputer.fit_transform(X_train_combined_data[non_missing_columns])

# Convert back to DataFrame with the correct column names
X_train_combined_data_imputed = pd.DataFrame(X_train_combined_data_imputed, columns=non_missing_columns)

# Ensure test_combined_data has the same columns as X_train_combined_data
test_combined_data_imputed = pd.DataFrame(
    imputer.transform(test_combined_data[non_missing_columns]),
    columns=non_missing_columns
)

# Apply PCA to find the top 10 features
n_components = 230
pca = PCA(n_components)
pca.fit(X_train_combined_data_imputed)

# Get the PCA components and feature importance
explained_variance = pca.explained_variance_ratio_
top_indices = np.argsort(-explained_variance)[:n_components]  # Get indices of the top components
top_features = X_train_combined_data_imputed.columns[top_indices]

# Use the top components for training
X_train_top_features = X_train_combined_data_imputed[top_features]
X_val_top_features = test_combined_data_imputed[top_features]

# Standardize features
scaler = StandardScaler()
X_train_top_features = scaler.fit_transform(X_train_top_features)
X_val_top_features = scaler.transform(X_val_top_features)

# Train-test split (use the same features)
X_train, X_val, Y_train, Y_val = train_test_split(
    X_train_top_features, Y_train_combined_data, test_size=0.2, stratify=Y_train_combined_data, random_state=1
)

# Define the objective function for Optuna
#def objective(trial):
#    # Suggest hyperparameters
#    C = trial.suggest_float('C', 1e-4, 1e2, log=True)  # Use suggest_float instead of suggest_loguniform
#    max_iter = trial.suggest_int('max_iter', 500, 1000)  # Maximum number of iterations
#    solver = trial.suggest_categorical('solver', ['lbfgs', 'liblinear', 'saga'])  # Optimization algorithm
#
#    # Initialize and train the logistic regression model
#    log_reg_model = LogisticRegression(
#        penalty='l2',
#        C=C,
#        solver=solver,
#        max_iter=max_iter,
#        random_state=42,
#        class_weight='balanced',
#        n_jobs=-1  # Use all CPU cores
#    )
#
#    # Fit the model
#    log_reg_model.fit(X_train, Y_train)
#
#    # Predict and evaluate on validation set
#    Y_pred_val = log_reg_model.predict(X_val)
#    f1_val = f1_score(Y_val, Y_pred_val)
#    
#    return f1_val  # Return the F1 score as the objective value
#
#
# 
# Create an Optuna study and optimize
#
#study = optuna.create_study(direction='maximize')
#study.optimize(objective, n_trials=50)  # Number of trials

# Print the best hyperparameters
#print("Best hyperparameters: ", study.best_params)
#print("Best F1 Score: ", study.best_value)

# Make predictions for the test set with the best parameters
#best_params = study.best_params

log_reg_model = LogisticRegression(
    penalty='l2',
    C=1e2,
    solver='liblinear',
    max_iter=1000,
    random_state=42,
    class_weight='balanced'
)

log_reg_model.fit(X_train, Y_train)

Y_submission = log_reg_model.predict(scaler.transform(test_combined_data_imputed[top_features]))



# Make predictions on training set
Y_pred_train = log_reg_model.predict(X_train)
# Make predictions on validation set
Y_pred_val = log_reg_model.predict(X_val)

# Calculate accuracy and F1 score for training set
accuracy_train = accuracy_score(Y_train, Y_pred_train)
f1_train = f1_score(Y_train, Y_pred_train)

# Calculate accuracy and F1 score for validation set
accuracy_val = accuracy_score(Y_val, Y_pred_val)
f1_val = f1_score(Y_val, Y_pred_val)

# Print results
print("Training Accuracy: ", accuracy_train)
print("Training F1 Score: ", f1_train)
#print("Validation Accuracy: ", accuracy_val)
#print("Validation F1 Score: ", f1_val)

# Creating a DataFrame for submission
submission = pd.DataFrame({
    'TransactionID': test_filtered_transaction_data['TransactionID'],  # Ensure this column exists
    'isFraud': Y_submission
})

# Save the predictions into a CSV file for submission
submission.to_csv('C:/Users/carlo/.vscode/Repos/Kaggle Models/Kaggle Data/IEEE-CIS Fraud Detection Data/submission.csv', index=False)
