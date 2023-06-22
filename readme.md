# Heart Disease Prediction

This project focuses on predicting the risk of heart disease based on various parameters. It utilizes logistic regression to train a model and make predictions. The dataset consists of parameters such as gender, age, education, smoking habits, blood pressure, cholesterol levels, BMI, and glucose levels.

## Dataset

The dataset used in this project is provided in the file `heart_disease.csv`. It contains the following parameters:

- Gender: Categorical (Male/Female)
- Age: Continuous
- Education: Categorical (Uneducated/Primary School/Other)
- Current Smoker: Categorical (0: No, 1: Yes)
- Cigarettes Per Day: Continuous
- BP Meds: Categorical (0: No, 1: Yes)
- Prevalent Stroke: Categorical (No/Yes)
- Prevalent Hypertension: Categorical (0: No, 1: Yes)
- Diabetes: Categorical (0: No, 1: Yes)
- Total Cholesterol: Continuous
- Systolic Blood Pressure: Continuous
- Diastolic Blood Pressure: Continuous
- BMI: Continuous
- Heart Rate: Continuous
- Glucose: Continuous
- Heart Stroke: Categorical (No/Yes)

## How it works

1. The dataset is loaded from the file `heart_disease.csv`. It contains information about various parameters and the occurrence of heart disease.

2. The dataset is preprocessed to handle missing values and encode categorical variables. Missing values in the 'glucose' column are imputed using the mean value. Categorical variables such as gender, education, prevalent stroke, and heart stroke are one-hot encoded.

3. Numeric variables are standardized using the StandardScaler class from scikit-learn. This ensures that all features have zero mean and unit variance, which can improve the performance of the logistic regression model.

4. The dataset is split into features (X) and the target variable (y). X contains all the parameters except 'Heart_ strokeyes' and 'Heart_ strokeNo', while y contains the 'Heart_ strokeyes' column indicating the occurrence of heart disease.

5. The dataset is further divided into train-validation and test sets using train_test_split from scikit-learn. The test set will be used to evaluate the final model.

6. The logistic regression model is trained using k-fold cross-validation on the train-validation set. The dataset is divided into k folds, and the model is trained on k-1 folds while evaluating the performance on the remaining fold. This process is repeated k times, and the average validation accuracy is calculated.

7. The model is then trained on the entire train-validation set using the optimal hyperparameters determined from the cross-validation process.

8. Finally, the trained model is evaluated on the test set. The test accuracy is calculated, and additional evaluation metrics such as the cost function, ROC curve, and confusion matrix are plotted.

## Requirements

- Python 3.x
- pandas
- numpy
- scikit-learn
- matplotlib


