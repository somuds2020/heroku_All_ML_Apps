# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
#from sklearn.ensemble import RandomForestClassifier
#from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn import metrics
from sklearn.pipeline import make_pipeline

data = pd.read_csv(r'C:\Users\SONY\Desktop\Data Science Practice\Resume_proj\Home\Data\Maison.csv')

# Since the columns are in french, in order to make them more readable, let's translate them into English
data = data.rename(index = str, columns = {'PRIX':'price','SUPERFICIE': 'area','CHAMBRES': 'rooms', 
                         'SDB': 'bathroom', 'ETAGES': 'floors','ALLEE': 'driveway',
                         'SALLEJEU':'game_room', 'CAVE': 'cellar', 
                         'GAZ': 'gas', 'AIR':'air', 'GARAGES': 'garage', 'SITUATION': 'situation'})

X = data[['area', 'rooms', 'bathroom', 'floors', 'driveway', 'game_room',
       'cellar', 'gas', 'air', 'garage', 'situation']]
y = data['price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

model = make_pipeline(StandardScaler(), LinearRegression())

model.fit(X_train,y_train)

y_pred_train=model.predict(X_train)
print('MAE :', metrics.mean_absolute_error(y_train, y_pred_train))
print('MSE :', metrics.mean_squared_error(y_train, y_pred_train))
print('RMSE :', np.sqrt(metrics.mean_squared_error(y_train, y_pred_train)))
print('R2 Score :', metrics.r2_score(y_train, y_pred_train))

y_pred_test=model.predict(X_test)
print('MAE :', metrics.mean_absolute_error(y_test, y_pred_test))
print('MSE :', metrics.mean_squared_error(y_test, y_pred_test))
print('RMSE :', np.sqrt(metrics.mean_squared_error(y_test, y_pred_test)))
print('R2 Score :', metrics.r2_score(y_test, y_pred_test))

# Saving model to disk
pickle.dump(model, open(r'C:\Users\SONY\Desktop\Data Science Practice\Resume_proj\Home\model_house.pkl','wb'))

# Loading model to compare the results
model = pickle.load(open(r'C:\Users\SONY\Desktop\Data Science Practice\Resume_proj\Home\model_house.pkl','rb'))
#print(model.predict([[6, 148, 72, 35, 0, 33.6, 0.625, 50]]))

test_x = np.array([[1, 85, 66, 29, 0, 26.6, 0.351, 31]])
print(model.predict(test_x))