# Set debug flag
debug = 1

import numpy as np  
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score
import seaborn as sns
from sklearn.model_selection import StratifiedKFold
import pandas as pd
import os
import tensorflow as tf

rootPath = 'C:/Users/carlo/.vscode/Repos/Kaggle Models/Kaggle Data/Titanic Data/'

# Debugging: Print the file paths in the dataset directory
for dirname, _, filenames in os.walk(rootPath):
    for filename in filenames:
        if debug == 1:
            print(os.path.join(dirname, filename))

    if debug == 1:
        print(dirname)

# Load dataset and set features/labels
trainDataset = pd.read_csv(rootPath +'train.csv')

# Fill missing values
trainDataset.loc[trainDataset['Age'].isnull(), 'Age'] = -1
trainDataset.loc[trainDataset['Embarked'].isnull(), 'Embarked'] = -1

# Separate features and labels
X = trainDataset.drop(columns=['Survived', 'Name', 'Ticket', 'Cabin'])
Y = trainDataset['Survived']

# Debugging: Print features and labels
if debug:
    print('X:')
    print(X.to_string(index=False))

X['Sex'] = X['Sex'].astype(str)
X['Embarked'] = X['Embarked'].astype(str)

# One-hot encoding
ct = ColumnTransformer(
    transformers=[
        ('encoder', OneHotEncoder(), [2, 7])
    ],
    remainder='passthrough'  # Leave the other columns as is
)

# Transform features
X_transformed = ct.fit_transform(X)

# Create a TensorFlow model
model = tf.keras.Sequential([
    tf.keras.layers.InputLayer(input_shape=X_transformed.shape[1]),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')  # Output layer for binary classification
])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Use Stratified K-Fold for cross-validation
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Initialize lists for accuracy and loss
accuracy_list = []
loss_list = []
val_accuracy_list = []
val_loss_list = []

all_predictions = []
all_true_values = []

for train_index, test_index in tqdm(kf.split(X_transformed, Y), total=kf.get_n_splits(), desc="Training Progress"):
    X_train, X_test = X_transformed[train_index], X_transformed[test_index]
    Y_train, Y_test = Y.iloc[train_index], Y.iloc[test_index]

    # Fit the model on the training data
    with tf.device('/GPU:0'):
        history = model.fit(X_train, Y_train, epochs=200, batch_size=16, 
                            verbose=0, validation_data=(X_test, Y_test))  # Add validation data here

    # Store average history for plotting later
    accuracy_list.append(history.history['accuracy'])
    val_accuracy_list.append(history.history['val_accuracy'])
    loss_list.append(history.history['loss'])
    val_loss_list.append(history.history['val_loss'])

    # Make predictions on the test data
    predictions = model.predict(X_test)
    predictions = (predictions > 0.5).astype(int)  # Convert probabilities to binary predictions
    
    # Store predictions and true values for this fold
    all_predictions.extend(predictions)
    all_true_values.extend(Y_test)

# Convert to NumPy arrays for easier handling
all_predictions = np.array(all_predictions).flatten()
all_true_values = np.array(all_true_values)

# Average the accuracy and loss across all folds
avg_train_accuracy = np.mean(accuracy_list, axis=0)
avg_val_accuracy = np.mean(val_accuracy_list, axis=0)
avg_train_loss = np.mean(loss_list, axis=0)
avg_val_loss = np.mean(val_loss_list, axis=0)

# Plot accuracy and loss
plt.figure(figsize=(12, 4))

# Accuracy plot
plt.subplot(1, 2, 1)
plt.plot(avg_train_accuracy, label='Train Accuracy', color='blue')
plt.plot(avg_val_accuracy, label='Validation Accuracy', color='orange')
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(loc='upper left')

# Loss plot
plt.subplot(1, 2, 2)
plt.plot(avg_train_loss, label='Train Loss', color='blue')
plt.plot(avg_val_loss, label='Validation Loss', color='orange')
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(loc='upper left')

plt.tight_layout()
plt.show()

print('avg train acc')
print(avg_train_accuracy)
print('avg val acc')
print(avg_val_accuracy)