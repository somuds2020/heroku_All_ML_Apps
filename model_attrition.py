# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler, LabelEncoder, OrdinalEncoder
#from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
#from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn import metrics
from sklearn.pipeline import Pipeline

data = pd.read_csv(r'C:\Users\SONY\Desktop\Data Science Practice\Resume_proj\Home\Data\HR_Employee_Attrition_Data.csv')

lb = LabelEncoder()
data['Attrition'] = lb.fit_transform(data['Attrition'])

oe = OrdinalEncoder()
data[['BusinessTravel', 'Department', 'EducationField', 'Gender','JobRole', 'MaritalStatus', 'Over18', 'OverTime']]=oe.fit_transform(data[['BusinessTravel', 'Department', 'EducationField', 'Gender','JobRole', 'MaritalStatus', 'Over18', 'OverTime']])

## Dummy Variable Coding
#data_coded = pd.get_dummies(data[data_cat].iloc[:,1:]) ##Sliced the Target Variable
#data_coded.columns

#Merging back Dummies dataset to original
#data_final = pd.concat([data[data_num],data_coded], axis=1)

#data_final['Attrition'] = data['Attrition'].map({"Yes":1, "No":0})

X = data.drop(['Attrition'],axis=1).values
y = data[['Attrition']].values.ravel()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

print(np.unique(y_train, return_counts=True))
print(np.unique(y_test, return_counts=True))

#Splitting Training and Test Set
#Since we have a very small dataset, we will train our model with all availabe data.

#model = Pipeline([('scaler', StandardScaler()),('logreg', LogisticRegression())])
model = RandomForestClassifier()
#model = SVC(gamma="auto")
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
pickle.dump(model, open(r'C:\Users\SONY\Desktop\Data Science Practice\Resume_proj\Home\model_attrition.pkl','wb'))
pickle.dump(oe, open(r'C:\Users\SONY\Desktop\Data Science Practice\Resume_proj\Home\label_attrition.pkl','wb'))

# Loading model to compare the results
model = pickle.load(open(r'C:\Users\SONY\Desktop\Data Science Practice\Resume_proj\Home\model_attrition.pkl','rb'))
label = pickle.load(open(r'C:\Users\SONY\Desktop\Data Science Practice\Resume_proj\Home\label_attrition.pkl','rb'))
#print(model.predict([[6, 148, 72, 35, 0, 33.6, 0.625, 50]]))

test_x = np.array([[41,"Travel_Rarely",1102,'Sales',1,2,'Life Sciences',1,1,2,'Female',94,3,2,'Sales Executive',4,'Single',5993,19479,8,'Y','Yes',11,3,1,80,0,8,0,1,6,4,0,5]])
#test_x = np.array([[49,'Travel_Frequently',279,'Research & Development',8,1,'Life Sciences',1,2,3,'Male',61,2,2,'Research Scientist',2,'Married',5130,24907,1,'Y','No',23,4,4,80,1,10,3,3,10,7,1,7]])
test_x_df = pd.DataFrame(test_x,columns=['Age','BusinessTravel','DailyRate','Department','DistanceFromHome','Education','EducationField','EmployeeCount','EmployeeNumber','EnvironmentSatisfaction','Gender','HourlyRate','JobInvolvement','JobLevel','JobRole','JobSatisfaction','MaritalStatus','MonthlyIncome','MonthlyRate','NumCompaniesWorked','Over18','OverTime','PercentSalaryHike','PerformanceRating','RelationshipSatisfaction','StandardHours','StockOptionLevel','TotalWorkingYears','TrainingTimesLastYear','WorkLifeBalance','YearsAtCompany','YearsInCurrentRole','YearsSinceLastPromotion','YearsWithCurrManager'])
    
test_x_df[['BusinessTravel', 'Department', 'EducationField', 'Gender','JobRole', 'MaritalStatus', 'Over18', 'OverTime']]=label.transform(test_x_df[['BusinessTravel', 'Department', 'EducationField', 'Gender','JobRole', 'MaritalStatus', 'Over18', 'OverTime']])
print(model.predict(test_x_df.values))