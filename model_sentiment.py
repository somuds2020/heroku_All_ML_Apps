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
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

data = pd.read_csv(r'C:\Users\SONY\Desktop\Data Science Practice\Resume_proj\Home\Data\Sentiment.csv', encoding = "ISO-8859-1")

X_train, X_test, y_train, y_test = train_test_split(data.SentimentText, data.Sentiment, test_size = 0.3, random_state = 42)

model = make_pipeline(CountVectorizer(stop_words='english',ngram_range=(1,2)), MultinomialNB())

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
pickle.dump(model, open(r'C:\Users\SONY\Desktop\Data Science Practice\Resume_proj\Home\model_sentiment.pkl','wb'))

# Loading model to compare the results
model = pickle.load(open(r'C:\Users\SONY\Desktop\Data Science Practice\Resume_proj\Home\model_sentiment.pkl','rb'))
#print(model.predict([[6, 148, 72, 35, 0, 33.6, 0.625, 50]]))

test_x = np.array(['the service sucks'])
print(model.predict(test_x))