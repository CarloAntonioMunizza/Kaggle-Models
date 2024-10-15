# Importing necessary libraries
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from FraudDetection import SetPaths, CreateDataFrames, TrainValTestSplit, LogRegInit, GenerateMetrics, GenerateSubmission

# Create basic paths
root_path, train_identity_path, train_transaction_path, test_identity_path, test_transaction_path = SetPaths()

# Create concatenated dataframes 
train_combined_data, test_combined_data = CreateDataFrames(train_identity_path, train_transaction_path, test_identity_path, test_transaction_path)

# Create train val test splits
X_train, X_val, Y_train, Y_val, X_test = TrainValTestSplit(train_combined_data, test_combined_data)

# Apply PCA
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
