# Import necessary libraries
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from FraudDetection import SetPaths, CreateDataFrames, LogRegInit, GenerateMetrics, GBXParamsInit, GenerateSubmission
import xgboost as xgb

# Create basic paths
root_path, train_identity_path, train_transaction_path, test_identity_path, test_transaction_path = SetPaths()

# Create concatenated dataframes
train_combined_data, test_combined_data = CreateDataFrames(train_identity_path, train_transaction_path, test_identity_path, test_transaction_path)

# Separate target and features
X_train_combined_data = train_combined_data.drop(['isFraud', 'TransactionID'], axis=1)
Y_train_combined_data = train_combined_data['isFraud']

# Extract TransactionID for submission purposes
transaction_ids = test_combined_data['TransactionID']

# Prepare test data without 'TransactionID'
X_test_combined_data = test_combined_data.drop('TransactionID', axis=1)

# Apply PCA
n_components = 370
pca = PCA(n_components)
X_train_pca = pca.fit_transform(X_train_combined_data)
X_test_pca = pca.transform(X_test_combined_data)

# Train-test split
X_train, X_val, Y_train, Y_val = train_test_split(
    X_train_pca, Y_train_combined_data, test_size=0.2, stratify=Y_train_combined_data, random_state=1
)

# Standardize features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test_pca)

# Train Logistic Regression model
xgb_model = xgb.XGBClassifier()
GBXParams = GBXParamsInit()

# Set the parameters for Grid Search
param_grid = {
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.1, 0.3],
    'n_estimators': [100, 200],
    'scale_pos_weight': [1, 5, 10]  # Adjust for class imbalance
}

# Perform grid search
grid_search = GridSearchCV(xgb_model, param_grid, scoring='f1', cv=3)
grid_search.fit(X_train, Y_train)

best_params = grid_search.best_params_

# Best parameters
print("Best parameters found: ", best_params)

# Convert datasets to DMatrix format
X_train_dmatrix = xgb.DMatrix(X_train, label=Y_train)
X_val_dmatrix = xgb.DMatrix(X_val, label=Y_val)
X_test_dmatrix = xgb.DMatrix(X_test)

# Train the model using the best parameters
num_boost_round = best_params['n_estimators']  # Set the number of boosting rounds from the best parameters
bst = xgb.train(best_params, X_train_dmatrix, num_boost_round=num_boost_round, evals=[(X_val_dmatrix, 'validation')], verbose_eval=True)

# Make predictions on training set
Y_pred_train = bst.predict(X_train_dmatrix)

# Convert probabilities to binary predictions
Y_pred_train_binary = [1 if pred > 0.5 else 0 for pred in Y_pred_train]

print('Train')
GenerateMetrics(Y_train, Y_pred_train_binary)

# Make predictions on validation set
Y_pred_val = bst.predict(X_val_dmatrix)
Y_pred_val_binary = [1 if pred > 0.5 else 0 for pred in Y_pred_val]

print('Val')
GenerateMetrics(Y_val, Y_pred_val_binary)

# After making predictions on the test set
Y_submission = bst.predict(X_test_dmatrix)

# Clean up the TransactionID values
transaction_ids = transaction_ids.astype(float).astype('Int32')  # Convert to float first, then to Int32

# Create submission DataFrame
submission = pd.DataFrame({
    'TransactionID': transaction_ids,  # Use the cleaned TransactionID
    'isFraud': [1 if pred > 0.5 else 0 for pred in Y_submission]
})

# Save the predictions into a CSV file for submission
submission.to_csv(root_path + 'submission.csv', index=False)
