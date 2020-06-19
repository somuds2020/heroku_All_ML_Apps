# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
#from sklearn.ensemble import RandomForestClassifier
#from sklearn.tree import DecisionTreeClassifier
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn import metrics
from sklearn.pipeline import make_pipeline

data = pd.read_csv(r'C:\Users\SONY\Desktop\Data Science Practice\Resume_proj\Home\Data\Cust_Spend_Data.csv')

X = data.drop(['Cust_ID','Name'], axis=1)

model = make_pipeline(StandardScaler(), KMeans(n_clusters=3, random_state=42))
#model = KMeans(n_clusters=3, random_state=42)
model.fit(X)

model.predict(X)

# Saving model to disk
pickle.dump(model, open(r'C:\Users\SONY\Desktop\Data Science Practice\Resume_proj\Home\model_cluster.pkl','wb'))

# Loading model to compare the results
model = pickle.load(open(r'C:\Users\SONY\Desktop\Data Science Practice\Resume_proj\Home\model_cluster.pkl','rb'))
#print(model.predict([[6, 148, 72, 35, 0, 33.6, 0.625, 50]]))

test_x = np.array([[1, 'A', 10000, 2, 1, 1, 0]])
print(model.predict(test_x[:,2:]))
