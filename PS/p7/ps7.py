import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC, SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix
import pathlib as pl 
import os

## DisableWarning: sklearn.exceptions.ConvergenceWarning
import warnings
from sklearn.exceptions import ConvergenceWarning
warnings.simplefilter(action='ignore', category=ConvergenceWarning)

script_dir = pl.Path(__file__).parent.absolute()
os.chdir(script_dir)

df = pd.read_csv('iris.csv')
print(df.head())

encoder = LabelEncoder()
df.Species = encoder.fit_transform(df.Species)

train , test = train_test_split(df, test_size=0.2, random_state=35)

train_x = train.drop(columns=['Species'],axis=1)
train_y = train['Species']

test_x = test.drop(columns=['Species'],axis=1)
test_y = test['Species']


models = {
    "Logistic Regression": LogisticRegression(),
    "SVM (one-vs-one)": SVC(),
    "Linear SVM (one-vs-rest)": LinearSVC(dual=False),
    "Decision Tree (Criterion : Entropy)": DecisionTreeClassifier(criterion='entropy')
}

for name, model in models.items():
    print(f"Model: {name}")
    model.fit(train_x,train_y)
    predict = model.predict(test_x)
    # actual_vs_predict = [f"{i} | {j}" for i,j in zip(encoder.inverse_transform(test_y),encoder.inverse_transform(predict))]
    # print('Actual vs Predicted')
    # for i in actual_vs_predict:
    #     print(i)
    print(f"Accuracy: {accuracy_score(test_y,predict)}")
    print(f"Confusion Matrix: \n{confusion_matrix(test_y,predict)}")