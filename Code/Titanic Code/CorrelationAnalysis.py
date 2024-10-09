import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load dataset and prepare it for analysis
rootPath = 'C:/Users/carlo/.vscode/Repos/Kaggle Models/Kaggle Data/Titanic Data/'
trainDataset = pd.read_csv(rootPath + 'train.csv')

# Fill missing values
trainDataset['Age'].fillna(trainDataset['Age'].mean(), inplace=True)
trainDataset['Fare'].fillna(trainDataset['Fare'].mean(), inplace=True)

# Convert categorical variables into numerical variables
trainDataset['Sex'] = trainDataset['Sex'].map({'male': 0, 'female': 1})
trainDataset['Embarked'].fillna('S', inplace=True)
trainDataset['Embarked'] = trainDataset['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})

# Select features for correlation analysis
features = ['Survived', 'Pclass', 'Sex', 'Age', 'Fare', 'SibSp', 'Parch', 'Embarked']

# Calculate correlation matrix
correlation_matrix = trainDataset[features].corr()

# Plot the heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title("Correlation Matrix of Titanic Features")
plt.show()
