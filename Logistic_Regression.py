import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Calculate the cost function
def cost_function(y, y_pred):
    return -np.mean(y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred))

# Calculate the gradients
def gradients(X, y, y_pred):
    m = len(y)
    return (1 / m) * np.dot(X.T, (y_pred - y))

# Update the weights
def update_weights(weights, learning_rate, grads):
    return weights - learning_rate * grads

data = pd.read_csv('heart_disease.csv')

def one_hot_encode(df, column):
    unique_values = df[column].unique()
    for value in unique_values:
        df[column+''+str(value)] = (df[column] == value).astype(int)
    df.drop(column, axis=1, inplace=True)
    return df

categorical_columns = ['Gender', 'education', 'prevalentStroke', 'Heart_ stroke']
for column in categorical_columns:
    data = one_hot_encode(data, column)

# Impute missing values in the 'glucose' column with its mean value
data['glucose'] = data['glucose'].fillna(data['glucose'].mean())

# Standardize numeric variables
numeric_columns = ['age', 'cigsPerDay', 'totChol', 'sysBP', 'diaBP', 'BMI', 'heartRate', 'glucose']
for column in numeric_columns:
    data[column] = (data[column] - data[column].mean()) / data[column].std()

# Separate features and target
X = data.drop(['Heart_ strokeyes', 'Heart_ strokeNo'], axis=1).values
y = data['Heart_ strokeyes'].values.reshape(-1, 1)

# Set the hyperparameters
learning_rate = 5
epochs = 100
k = 5  # Number of folds
test_size = 0.2  # Proportion of data to be used for the test set

# Perform train-validation-test split
X_trainval, X_test, y_trainval, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

imputer = SimpleImputer(strategy='mean')
X_trainval = pd.DataFrame(imputer.fit_transform(X_trainval))
X_test = pd.DataFrame(imputer.fit_transform(X_test))

# Perform k-fold cross-validation on the train-validation set
fold_size = len(X_trainval) // k
accuracy_scores = []

for i in range(k):
    # Split train-validation set into training and validation sets
    start = i * fold_size
    end = (i + 1) * fold_size
    X_val = X_trainval[start:end]
    y_val = y_trainval[start:end]
    X_train = np.concatenate((X_trainval[:start], X_trainval[end:]), axis=0)
    y_train = np.concatenate((y_trainval[:start], y_trainval[end:]), axis=0)

    # Add an intercept term (column of 1's) to the features matrix
    X_train = np.hstack([np.ones((X_train.shape[0], 1)), X_train])
    X_val = np.hstack([np.ones((X_val.shape[0], 1)), X_val])

    imputer = SimpleImputer(strategy='mean')
    X_train = pd.DataFrame(imputer.fit_transform(X_train))
    X_val = pd.DataFrame(imputer.transform(X_val))

    # Initialize the weights
    weights = np.zeros((X_train.shape[1], 1))

    # Train the model
    cost_list = []
    for epoch in range(epochs):
        z = np.dot(X_train, weights)
        y_pred_train = sigmoid(z)
        cost = cost_function(y_train, y_pred_train)
        grads = gradients(X_train, y_train, y_pred_train)
        weights = update_weights(weights, learning_rate, grads)
        cost_list.append(cost)

    # Make predictions on the validation set
    predictions_val = sigmoid(np.dot(X_val, weights))
    y_pred_val = (predictions_val > 0.5).astype(int)

    # Calculate accuracy on the validation set
    accuracy_val = np.mean(y_pred_val == y_val)
    accuracy_scores.append(accuracy_val)

# Calculate the average accuracy across all folds
average_accuracy = np.mean(accuracy_scores)
print(f"Average Validation Accuracy: {average_accuracy}")

# Train the final model on the entire train-validation set
X_trainval = np.hstack([np.ones((X_trainval.shape[0], 1)), X_trainval])
X_test = np.hstack([np.ones((X_test.shape[0], 1)), X_test])

imputer = SimpleImputer(strategy='mean')
X_trainval = pd.DataFrame(imputer.fit_transform(X_trainval))
X_test = pd.DataFrame(imputer.transform(X_test))

weights = np.zeros((X_trainval.shape[1], 1))

cost_list = []
for epoch in range(epochs):
    z = np.dot(X_trainval, weights)
    y_pred_trainval = sigmoid(z)
    cost = cost_function(y_trainval, y_pred_trainval)
    grads = gradients(X_trainval, y_trainval, y_pred_trainval)
    weights = update_weights(weights, learning_rate, grads)
    cost_list.append(cost)

# Make predictions on the test set
predictions_test = sigmoid(np.dot(X_test, weights))
y_pred_test = (predictions_test > 0.5).astype(int)

# Calculate accuracy on the test set
accuracy_test = np.mean(y_pred_test == y_test)
print(f"Test Accuracy: {accuracy_test}")

import matplotlib.pyplot as plt
epochs_array = np.arange(epochs)
plt.plot(epochs_array, cost_list)
plt.xlabel('Epochs')
plt.ylabel('Cost')
plt.title('Cost Function over Epochs')
plt.show()

from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix, classification_report
fpr, tpr, thresholds = roc_curve(y_test, predictions_test)
plt.plot(fpr, tpr, label='ROC curve')
plt.plot([0, 1], [0, 1], linestyle='--', label='Random Classifier')
plt.xlabel('False Positive Rate (FPR)')
plt.ylabel('True Positive Rate (TPR)')
plt.title('ROC Curve')
plt.legend()
plt.show()
# print(y_pred_test)