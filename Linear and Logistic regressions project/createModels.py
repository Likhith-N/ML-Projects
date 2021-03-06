import numpy as np
import seaborn as sns
from sklearn import datasets
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.linear_model import LogisticRegression
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics
import math

# Reading Dataset file
df = pd.read_csv(
    "./ISL-linear-regression-master/data/Advertising.csv")

# Dividing the data to train the model
x = df.iloc[:, 1:-1].values
y = df.iloc[:, 4].values

# creating trainng and testing dataset
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.20, random_state=0)

# creating best fit line and predicting the test data
lr = LinearRegression()
lr.fit(x_train, y_train)
y_pred = lr.predict(x_test)

# output : MAE,MSE & RMSE
print("MAE =", format(metrics.mean_absolute_error(y_test, y_pred)))
print("MSE =", format(metrics.mean_squared_error(y_test, y_pred)))
print("RMSE =", format(metrics.mean_squared_error(y_test, y_pred, squared=False)))

# Reading Dataset file
df1 = pd.read_csv(
    "./bike sharing data/train.csv")

#col_names = ["season", "holiday", "workingday","weather", "temp", "atemp", "humidity", "windspeed"]
# Dividing the data to train the model
x = df1.iloc[:, 1:9].values
y = df1.iloc[:, 11].values

# splitting the data
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.02, random_state=0)

# creating best fit line and predicting the test data
lr = LinearRegression()
lr.fit(x_train, y_train)
y_pred = lr.predict(x_test)

# output : MSE
print("MSE =", format(metrics.mean_squared_error(y_test, y_pred)))

# ploting graph actual vs predicted values
plt.bar(y_test, y_pred, width=1)
plt.title("Actual values vs Prected values")
plt.xlabel("Actual Values")
plt.ylabel("Predicted values")
plt.grid()
plt.show()

# Logistic Regression model


def LogReg(x, y, lg):
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.15, random_state=0)
    lg.fit(x_train, y_train)
    y_pred = lg.predict(x_test)
    conf_matrix = metrics.confusion_matrix(y_test, y_pred)

    print("\n Values of actual valued dataset :")
    print("\n Accuracy Score : ", format(
        metrics.accuracy_score(y_test, y_pred)))
    print("Confusion Matrix : \n", format(conf_matrix))

# Logistic Regression model of Scaled data


def Scale_LogReg(x, y, lg):
    columns = x.columns
    scaler = StandardScaler()
    std_x = scaler.fit_transform(x)
    std_x = pd.DataFrame(std_x, columns=columns)
    x_train, x_test, y_train, y_test = train_test_split(
        std_x, y, test_size=0.15, random_state=0)
    lg.fit(x_train, y_train)
    y_pred = lg.predict(x_test)
    conf_matrix = metrics.confusion_matrix(y_test, y_pred)

    print("\n Values of Scaled dataset :")
    print("\n Accuracy Score : ", format(
        metrics.accuracy_score(y_test, y_pred)))
    print("Confusion Matrix : \n ", format(conf_matrix))


# Logistic Regression model of Normalized data
def norm_LogReg(x, y, lg):
    columns = x.columns
    norm = MinMaxScaler()
    norm_x = norm.fit_transform(x)
    norm_x = pd.DataFrame(norm_x, columns=columns)
    x_train, x_test, y_train, y_test = train_test_split(
        norm_x, y, test_size=0.15, random_state=0)
    lg.fit(x_train, y_train)
    y_pred = lg.predict(x_test)
    conf_matrix = metrics.confusion_matrix(y_test, y_pred)

    print("\n Values of Normalized dataset :")
    print("\n Accuracy Score : ", format(
        metrics.accuracy_score(y_test, y_pred)))
    print("Confusion Matrix : \n ", format(conf_matrix))


# Lodaing the cancer dataset
cancer = datasets.load_breast_cancer()

# Extracting only the columns and rows from the loaded cancer data file
data = pd.DataFrame(cancer.data, columns=[cancer.feature_names])
data['Target'] = pd.Series(data=cancer.target, index=data.index)

# dividing data into x and y
x = data.drop('Target', axis=1)
y = data.Target.values
lg = LogisticRegression()

LogReg(x, y, lg)
Scale_LogReg(x, y, lg)
norm_LogReg(x, y, lg)
