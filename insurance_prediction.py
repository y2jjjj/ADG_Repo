import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics

# Load data
df = pd.read_csv('insurance.csv')

print("Data loaded successfully!")
print(df.head())

# Dataset info
print("\nDataset Info:")
df.info()

# Statistical summary
print("\nDataset Description:")
print(df.describe())

# Missing values
print("\nMissing Values:")
print(df.isnull().sum())

# Visualizations
sns.set()

# Age distribution
plt.figure(figsize=(6,6))
sns.distplot(df['age'])
plt.title('Age Distribution')
plt.show()

# Sex distribution
plt.figure(figsize=(6,6))
sns.countplot(x='sex', data=df)
plt.title('Sex Distribution')
plt.show()

# BMI distribution
plt.figure(figsize=(6,6))
sns.distplot(df['bmi'])
plt.title('BMI Distribution')
plt.show()

# Children distribution
plt.figure(figsize=(6,6))
sns.countplot(x='children', data=df)
plt.title('Children')
plt.show()

print(df['children'].value_counts())

# Smoker distribution
plt.figure(figsize=(6,6))
sns.countplot(x='smoker', data=df)
plt.title('Smoker')
plt.show()

print(df['smoker'].value_counts())

# Region distribution
plt.figure(figsize=(6,6))
sns.countplot(x='region', data=df)
plt.title('Region')
plt.show()

print(df['region'].value_counts())

# Charges distribution
plt.figure(figsize=(6,6))
sns.distplot(df['charges'])
plt.title('Charges Distribution')
plt.show()

# Encoding
df.replace({'sex':{'male':0,'female':1}}, inplace=True)
df.replace({'smoker':{'yes':0,'no':1}}, inplace=True)
df.replace({'region':{'southeast':0,'southwest':1,'northeast':2,'northwest':3}}, inplace=True)

X = df.drop(columns='charges', axis=1)
y = df['charges']

print(X)
print(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2)
print(X.shape, X_train.shape, X_test.shape)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Train predictions
train_preds = model.predict(X_train)

# Train R2
r2_train = metrics.r2_score(y_train, train_preds)
print('R-squared (train):', r2_train)

# Test predictions
test_preds = model.predict(X_test)

# Test R2
r2_test = metrics.r2_score(y_test, test_preds)
print('R-squared (test):', r2_test)

# Calculate evaluation metrics for training set
train_mae = metrics.mean_absolute_error(y_train, train_preds)
train_mse = metrics.mean_squared_error(y_train, train_preds)
train_rmse = np.sqrt(train_mse)

# Calculate evaluation metrics for test set
test_mae = metrics.mean_absolute_error(y_test, test_preds)
test_mse = metrics.mean_squared_error(y_test, test_preds)
test_rmse = np.sqrt(test_mse)

print('Training Set Evaluation Metrics:')
print('Mean Absolute Error:', train_mae)
print('Mean Squared Error:', train_mse)
print('Root Mean Squared Error:', train_rmse)

print('Test Set Evaluation Metrics:')
print('Mean Absolute Error:', test_mae)
print('Mean Squared Error:', test_mse)
print('Root Mean Squared Error:', test_rmse)

# Identify the lowest value of loss
lowest_loss = min(train_mae, train_mse, train_rmse, test_mae, test_mse, test_rmse)
print('Lowest value of loss (error) achieved:', lowest_loss)

# Sample prediction
input_data = (31,1,25.74,0,1,0)
input_array = np.asarray(input_data).reshape(1,-1)

prediction = model.predict(input_array)
print('Predicted insurance cost: USD', prediction[0])
