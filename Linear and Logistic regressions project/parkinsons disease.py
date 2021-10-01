# Importing necessary libraries and its methods
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn import metrics
import seaborn as sns

# Logistic Regression model of Scaled data


def LogReg(x, y, lg, i):

    # Scaling the dataset thrugh standardization
    columns = x.columns
    Icolumns = i.columns
    scaler = MinMaxScaler()
    std_x = scaler.fit_transform(x)
    std_x = pd.DataFrame(std_x, columns=columns)
    Input = scaler.fit_transform(i)
    Input = pd.DataFrame(Input, columns=Icolumns)
    print("\nThe Scaled / Normalized dataset : ")
    print(std_x.head())

    # Spliting the datasets for training and testing
    x_train, x_test, y_train, y_test = train_test_split(
        std_x, y, test_size=0.05, random_state=6)

    # Training and building model
    lg.fit(x_train, y_train)
    classifier = KNeighborsClassifier(n_neighbors=12)
    classifier.fit(x_train, y_train)
    K_pred = classifier.predict(x_test)
    L_pred = lg.predict(x_test)

    # Confusion matrix
    conf_matrix_Lg = metrics.confusion_matrix(y_test, L_pred)
    conf_matrix_Kn = metrics.confusion_matrix(y_test, K_pred)

    print("\n Values :")
    print("\n Accuracy Scores : ")
    print("\n Logistic Reggresion : ", format(
        metrics.accuracy_score(y_test, L_pred)))
    print("\n \t Confusion Matrix : \n \t ", format(conf_matrix_Lg))
    print("\n K Nearest Neighbors : ", format(
        metrics.accuracy_score(y_test, K_pred)))
    print("\n \t Confusion Matrix : \n \t ", format(conf_matrix_Kn))

    # Predicting the outcome of the user's data
    I_pred = lg.predict(Input)
    print("\n User input's predicted output : \n")
    print(I_pred)


# Reading csv file of the parkinsons disease dataset and dividing the x and y data
df = pd.read_csv("parkinsons.csv")
x = df.drop(['status', 'name'], axis=1)
y = df['status']

userInput = pd.read_csv(
    "C:/Users/likic/OneDrive/Desktop/tequed files/Tequed labs Project/UserInput.csv")


# Identifying the null/empty values in the file for data cleaning
print("Info of the data file: \n")
print(df.info())
print("\nData cleaning : Checking if there are any null values (0 represents no null values in the column) : \n")
print(x.isnull().sum())

lg = LogisticRegression()

LogReg(x, y, lg, userInput)

df.hist()
# plt.bar(df[MDVP:Fo(Hz)],df[status], width = 0.6)
# plt.xlabel("MDVP:Fo(Hz)")
# plt.ylabel("Status")
plt.show()
