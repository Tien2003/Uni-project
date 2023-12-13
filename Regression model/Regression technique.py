import pandas as pd
from pandas import read_csv
import numpy as np
from sklearn.impute import KNNImputer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.tree import DecisionTreeRegressor
import matplotlib.pyplot as plt

# Load training and testing datasets
train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')

#replacing the 'Missing' values with NaN
train_data = train_data.replace('Missing', np.nan)
test_data = test_data.replace('Missing', np.nan)

# Identify numeric columns
numeric_columns = train_data.select_dtypes(include='number').columns

#imputing missing values using KNN

#Spliting train_data into features and target variable
X_train = train_data[numeric_columns].drop(['SalePrice','ID'], axis=1)
y_train = train_data['SalePrice']

#Spliting test_data
X_test = test_data[numeric_columns].drop(['SalePrice','ID'], axis=1)
y_test = test_data['SalePrice']


#Create KNN imputer
imputer = KNNImputer(n_neighbors=3)

#Impute missing values
X_train_knn = imputer.fit_transform(X_train)
X_test_knn = imputer.transform(X_test)

#train linear regression model
model = LinearRegression()
model.fit(X_train_knn, y_train)

#Prediction on the test set
y_pred = model.predict(X_test_knn)

# Assess the model's generalization performance
rmse_lr = np.sqrt(mean_squared_error(y_test, y_pred))

# rmse_lr

#train Decision Tree Regressor model
model = DecisionTreeRegressor(random_state=40)
model.fit(X_train_knn, y_train)

#Prediction on the test set
y_pred = model.predict(X_test_knn)

# Assess the model's generalization performance
rmse_dt = np.sqrt(mean_squared_error(y_test, y_pred))
# rmse_dt

#display the DataFrame results 
results = pd.DataFrame({
    'Method': ['Linear Regression', 'Decision Tree Regression'],
    'RMSE': [rmse_lr, rmse_dt]
})
print(results)

# Create a bar plot to visually compare the RMSE values
plt.figure(figsize=(6, 3))
plt.bar(results['Method'], results['RMSE'], color=['pink', 'black'])
plt.ylabel('RMSE')
plt.show()