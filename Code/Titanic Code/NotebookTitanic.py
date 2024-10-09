import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import optuna

# Load datasets
trainDataset = pd.read_csv('C:/Users/carlo/.vscode/Repos/Kaggle Models/Kaggle Data/Titanic Data/train.csv')
testDataset = pd.read_csv('C:/Users/carlo/.vscode/Repos/Kaggle Models/Kaggle Data/Titanic Data/test.csv')

# Handle missing values
trainDataset['Age'] = trainDataset['Age'].fillna(-1)
trainDataset['Embarked'] = trainDataset['Embarked'].fillna(-1)

# Add Family Size feature
trainDataset['FamilySize'] = trainDataset['SibSp'] + trainDataset['Parch'] + 1

# Select features
X = trainDataset[['Sex', 'Pclass', 'Fare', 'FamilySize']]
Y = trainDataset['Survived']

# Convert 'Sex' to string for one-hot encoding
X.loc[:, 'Sex'] = X['Sex'].astype(str)

# One-hot encode 'Sex'
ct = ColumnTransformer(
    transformers=[('encoder', OneHotEncoder(), ['Sex'])],
    remainder='passthrough'
)

# Standardize features
scaler = StandardScaler()

# Stratified K-Fold cross-validation (3 folds)
kf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

# Define the objective function for Optuna
def objective(trial):
    n_estimators = trial.suggest_int('n_estimators', 50, 200)
    max_depth = trial.suggest_int('max_depth', 1, 10)
    min_samples_split = trial.suggest_int('min_samples_split', 2, 10)
    min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 10)

    # Correcting max_features options
    max_features = trial.suggest_categorical('max_features', ['sqrt', 'log2', None])

    bootstrap = trial.suggest_categorical('bootstrap', [True, False])

    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        max_features=max_features,
        bootstrap=bootstrap,
        random_state=42
    )


    scores = []
    for train_index, test_index in kf.split(X, Y):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        Y_train, Y_test = Y.iloc[train_index], Y.iloc[test_index]

        # Transform features using ColumnTransformer
        X_train_transformed = ct.fit_transform(X_train)
        X_test_transformed = ct.transform(X_test)

        # Scale the features
        X_train_scaled = scaler.fit_transform(X_train_transformed)
        X_test_scaled = scaler.transform(X_test_transformed)

        # Train the model
        model.fit(X_train_scaled, Y_train)

        # Make predictions
        predictions = model.predict(X_test_scaled)

        # Calculate accuracy
        accuracy = accuracy_score(Y_test, predictions)
        scores.append(accuracy)

    return np.mean(scores)

# Create Optuna study and optimize
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=4000)

# Get the best hyperparameters
best_params = study.best_params
print('Best parameters found: ', best_params)

# Train the model with the best parameters on the entire dataset
best_model = RandomForestClassifier(**best_params)

# Transform and scale the entire dataset
X_transformed = ct.fit_transform(X)
X_scaled = scaler.fit_transform(X_transformed)

# Train the best model
best_model.fit(X_scaled, Y)

# --- Predictions on the test dataset ---
# Handle missing values in the test set
testDataset['Age'] = testDataset['Age'].fillna(-1)
testDataset['Fare'] = testDataset['Fare'].fillna(testDataset['Fare'].median())
testDataset['FamilySize'] = testDataset['SibSp'] + testDataset['Parch'] + 1

# Select features for the test dataset
X_test_final = testDataset[['Sex', 'Pclass', 'Fare', 'FamilySize']]
X_test_final.loc[:, 'Sex'] = X_test_final['Sex'].astype(str)

# Transform and scale the test data
X_test_transformed_final = ct.transform(X_test_final)
X_test_scaled_final = scaler.transform(X_test_transformed_final)

# Make predictions on the test dataset
test_predictions = best_model.predict(X_test_scaled_final)

# Create a DataFrame with PassengerId and the predicted Survived values
prediction_output = pd.DataFrame({
    'PassengerId': testDataset['PassengerId'],
    'Survived': test_predictions.astype(int)
})

# Save to CSV
prediction_output.to_csv('C:/Users/carlo/.vscode/Repos/Kaggle Models/Kaggle Data/Titanic Data/submission.csv', index=False)

print("Prediction CSV saved as 'submission.csv'.")
