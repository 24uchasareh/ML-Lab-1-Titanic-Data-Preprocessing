import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("titanic.csv")
df.head()

df.tail()
print(df.shape)
df.shape
df.columns

df.info()
df.describe()

df.isnull().sum()
df.head()
df["Age"].fillna(df["Age"].median(), inplace=True)
df["Embarked"].fillna(df["Embarked"].mode()[0], inplace=True)
df.drop("Cabin", axis=1, inplace=True)

df.head()
df.duplicated().sum()
sns.boxplot(x=df["Age"])
plt.show()

sns.boxplot(x=df["Fare"])
plt.show()

Q1 = df["Age"].quantile(0.25)
Q3 = df["Age"].quantile(0.75)
IQR = Q3 - Q1
lower = Q1 - 1.5*IQR
upper = Q3 + 1.5*IQR
df = df[(df["Age"] >= lower) & (df["Age"] <= upper)]

sns.boxplot(x=df["Age"])
plt.show()

Q1 = df["Fare"].quantile(0.25)
Q3 = df["Fare"].quantile(0.75)
IQR = Q3 - Q1
lower = Q1 - 1.5*IQR
upper = Q3 + 1.5*IQR
df = df[(df["Fare"] >= lower) & (df["Fare"] <= upper)]

sns.boxplot(x=df["Fare"])
plt.show()

df["Sex"] = df["Sex"].replace("male", 0)
df["Sex"] = df["Sex"].replace("female", 1)
df.head()

df["Embarked"] = df["Embarked"].replace("S", 0)
df["Embarked"] = df["Embarked"].replace("C", 1)
df["Embarked"] = df["Embarked"].replace("Q", 2)
df.head()

plt.figure(figsize=(6,4))
sns.histplot(df["Age"], bins = 20, kde=True)
plt.title("Age disribution")
plt.show()

plt.figure(figsize=(6,4))
sns.histplot(df["Fare"], bins = 20, kde=True)
plt.title("Fare disribution")
plt.show()

sns.countplot(x="Sex", data=df)
plt.title("Gender Count")
plt.show()

sns.countplot(x="Sex", hue = "Survived", data=df)
plt.title("Gender vs Survived")
plt.show()

sns.countplot(x="Pclass", hue = "Survived", data=df)
plt.title("Pclass vs Survived")
plt.show()

sns.scatterplot(x="Age", y = "Fare", data=df)
plt.title("Age vs Fare")
plt.show()

plt.figure(figsize=(10,8))
numeric_df = df.select_dtypes(include=['number'])
sns.heatmap(numeric_df.corr(),
    annot=True,
    cmap='magma')
plt.show()

#Feature Scaling
x = df.drop("Survived", axis=1)
y = df["Survived"]
x = x.drop(["PassengerId", "Name", "Ticket"], axis=1)

#Standard Scaling
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
x[["Age", "Fare"]]= scaler.fit_transform(x[["Age", "Fare"]])
x.head(15)

#Train Test Split
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.20,random_state=42)
print("Training Data: ", x_train.shape)
print("Testing Data: ", x_test.shape)
print(x_train.head())

import warnings
warnings.filterwarnings("ignore")
