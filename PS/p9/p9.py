import pathlib as pl 
import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import cross_val_score


script_dir = pl.Path(__file__).parent.absolute()
os.chdir(script_dir)

df = pd.read_csv('iris.csv')

feature_columns = ['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm','PetalWidthCm']
X = df[feature_columns].values
y = df['Species'].values

encoder = LabelEncoder()
y = encoder.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=35)


sns.pairplot(df.drop("Id", axis=1), hue='Species', height=2, markers=['o', 's', 'D'])

classifier = KNeighborsClassifier(n_neighbors=3)
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)

cm = confusion_matrix(y_test, y_pred)
print(f"Confusion Matrix: \n{cm}")

accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")


plt.show()
input()