# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn import metrics
from sklearn.pipeline import Pipeline

data_train_sample = pd.read_csv(r'C:\Users\SONY\Desktop\Data Science Practice\Resume_proj\Home\Data\train_loan.csv')
data_test = pd.read_csv(r'C:\Users\SONY\Desktop\Data Science Practice\Resume_proj\Home\Data\test_loan.csv')

data_over = data_train_sample[data_train_sample.Approved == 1]
data_under = data_train_sample[data_train_sample.Approved == 0].iloc[:22000,:]

data_train = pd.concat([data_under,data_over,data_over,data_over,data_over,data_over,data_over,data_over,data_over,data_over,data_over])
data_train.Approved.value_counts()

## Create Age column

data_train['DOB'] = pd.to_datetime(data_train['DOB'], format='%d/%m/%y')
data_train['Age'] = 2020 - data_train.DOB.dt.year

data_test['DOB'] = pd.to_datetime(data_test['DOB'], format='%d/%m/%y')
data_test['Age'] = 2020 - data_test.DOB.dt.year

data_train.isnull().sum()
data_test.isnull().sum()

X = data_train.drop(['Approved','DOB','ID','City_Code','Employer_Code','Source','Customer_Existing_Primary_Bank_Code','Lead_Creation_Date'], axis=1)
y = data_train[['Approved']]
data_test = data_test.drop(['DOB','ID','City_Code','Employer_Code','Source','Customer_Existing_Primary_Bank_Code','Lead_Creation_Date'], axis=1)

X.fillna(X.mean(), inplace=True)
data_test.fillna(data_test.mean(), inplace=True)

data_car = X.select_dtypes(include=np.object).columns
data_car

for i in data_car:
    X[i].fillna(X[i].mode()[0], inplace=True)
    data_test[i].fillna(data_test[i].mode()[0], inplace=True)

X.isnull().sum()
data_test.isnull().sum()

oe = OrdinalEncoder()
X[data_car]=oe.fit_transform(X[data_car])
data_test[data_car]=oe.transform(data_test[data_car])

oe.categories_

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 42)

print(np.unique(y_train, return_counts=True))
print(np.unique(y_test, return_counts=True))

#Splitting Training and Test Set
#Since we have a very small dataset, we will train our model with all availabe data.

#model = Pipeline([('scaler', StandardScaler()),('logreg', LogisticRegression())])
#model = DecisionTreeClassifier(criterion = "gini", random_state = 100,max_depth=10, min_samples_leaf=5)
model = RandomForestClassifier()
model.fit(X_train.values, y_train.values.ravel())

y_pred_train=model.predict(X_train)
print("Accuracy:",metrics.accuracy_score(y_train, y_pred_train))
print("Precision:",metrics.precision_score(y_train, y_pred_train))
print("Recall:",metrics.recall_score(y_train, y_pred_train))

y_pred_test=model.predict(X_test)
print("Accuracy:",metrics.accuracy_score(y_test, y_pred_test))
print("Precision:",metrics.precision_score(y_test, y_pred_test))
print("Recall:",metrics.recall_score(y_test, y_pred_test))

y_pred_val = model.predict(data_test)

# Saving model to disk
pickle.dump(model, open(r'C:\Users\SONY\Desktop\Data Science Practice\Resume_proj\Home\model_loan.pkl','wb'))
pickle.dump(oe, open(r'C:\Users\SONY\Desktop\Data Science Practice\Resume_proj\Home\label_loan.pkl','wb'))
