# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
#from sklearn.ensemble import RandomForestClassifier
#from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn import metrics
from sklearn.pipeline import Pipeline

data = pd.read_csv(r'C:\Users\SONY\Desktop\Data Science Practice\Resume_proj\diabetes\diabetes.csv')

data[['Glucose','BMI']] = data[['Glucose','BMI']].replace(0, np.NaN)
data.dropna(inplace=True)

X = data[['Pregnancies','Glucose','BloodPressure','SkinThickness','Insulin','BMI',
          'DiabetesPedigreeFunction','Age']].values
y = data[['Outcome']].values.ravel()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

print(np.unique(y_train, return_counts=True))
print(np.unique(y_test, return_counts=True))

#Splitting Training and Test Set
#Since we have a very small dataset, we will train our model with all availabe data.

#model = Pipeline([('scaler', StandardScaler()),('logreg', LogisticRegression())])
model = LogisticRegression()
model.fit(X_train, y_train)

y_pred_train=model.predict(X_train)
print("Accuracy:",metrics.accuracy_score(y_train, y_pred_train))
print("Precision:",metrics.precision_score(y_train, y_pred_train))
print("Recall:",metrics.recall_score(y_train, y_pred_train))

y_pred_test=model.predict(X_test)
print("Accuracy:",metrics.accuracy_score(y_test, y_pred_test))
print("Precision:",metrics.precision_score(y_test, y_pred_test))
print("Recall:",metrics.recall_score(y_test, y_pred_test))

# Saving model to disk
pickle.dump(model, open(r'C:\Users\SONY\Desktop\Data Science Practice\Resume_proj\Home\model_diabetes.pkl','wb'))

# Loading model to compare the results
model = pickle.load(open(r'C:\Users\SONY\Desktop\Data Science Practice\Resume_proj\Home\model_diabetes.pkl','rb'))
#print(model.predict([[6, 148, 72, 35, 0, 33.6, 0.625, 50]]))

test_x = np.array([[1, 85, 66, 29, 0, 26.6, 0.351, 31]])
print(model.predict(test_x))