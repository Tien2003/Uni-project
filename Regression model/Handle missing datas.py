import pandas as pd
from pandas import read_csv
import numpy as np
from sklearn.impute import KNNImputer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import seaborn as sns


train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')

# count =0
# for i in train_data.columns:
#     count =0
#     for j in train_data[i]:
#         if j == "Missing": 
#             count+=1
# #    print(count)
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

# count_grvl = (train_data['Street'] == 'Grvl').sum()
# count_grvl


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
imputer = KNNImputer(n_neighbors=9)

#Impute missing values
X_train_knn = imputer.fit_transform(X_train)
X_test_knn = imputer.transform(X_test)

#train linear regression model
model = LinearRegression()
model.fit(X_train_knn, y_train)

#prediction
y_pred = model.predict(X_test_knn)

# Assess the model's generalization performance using R-quared
r2_knn = r2_score(y_test, y_pred)


#Handling missing values by removing the rows have missing values
#Deleting the missing data
train_cleaned = train_data.dropna()
test_cleaned = test_data.dropna()

#Spliting train_data into features and target variable
X_train = train_cleaned[numeric_columns].drop(['SalePrice','ID'], axis=1)
y_train = train_cleaned['SalePrice']

#Spliting test_data
X_test = test_cleaned[numeric_columns].drop(['SalePrice','ID'], axis=1)
y_test = test_cleaned['SalePrice']

#train linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

#prediction
y_pred = model.predict(X_test)

# Assess the model's generalization performance using R-quared
r2_rmv = r2_score(y_test, y_pred)

