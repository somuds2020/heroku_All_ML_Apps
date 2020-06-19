import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)
model_diabetes = pickle.load(open('model_diabetes.pkl', 'rb'))
model_attrition = pickle.load(open('model_attrition.pkl', 'rb'))
label_attrition = pickle.load(open('label_attrition.pkl', 'rb'))
model_cluster = pickle.load(open('model_cluster.pkl', 'rb'))
model_house = pickle.load(open('model_house.pkl', 'rb'))
model_sentiment = pickle.load(open('model_sentiment.pkl', 'rb'))
model_loan = pickle.load(open('model_loan.pkl', 'rb'))
label_loan = pickle.load(open('label_loan.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('home.html')
    
@app.route('/diabetes',methods=['POST'])
def diabetes():
    return render_template('diabetes.html')

@app.route('/diabetes/predict',methods=['POST'])
def predict_diabetes():
    '''
    For rendering results on HTML GUI
    '''
    int_features = [float(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    
    prediction = model_diabetes.predict(final_features)

    if prediction == 1:
        output = "Diabetic"
    else:
        output = "Non Diabetic"

    return render_template('diabetes.html', prediction_text='Person is {}'.format(output))

@app.route('/attrition',methods=['POST'])
def attrition():
    return render_template('attrition.html')
    
@app.route('/attrition/predict',methods=['POST'])
def predict_attrition():
    '''
    For rendering results on HTML GUI
    '''
    features = [x for x in request.form.values()]
    final_features = []
    
    # Need to convert features into numeric and string

    for i,v in enumerate(features):
        if (i == 1 or i == 3 or i == 6 or i == 10 or i == 14 or i == 16 or i == 20 or i == 21):
            final_features.append(v)
        else:
    	    final_features.append(int(v))
    
    # Creating dataframe for ordinal encoding
    attrition_df = pd.DataFrame([final_features], columns=['Age','BusinessTravel','DailyRate','Department','DistanceFromHome','Education','EducationField','EmployeeCount','EmployeeNumber','EnvironmentSatisfaction','Gender','HourlyRate','JobInvolvement','JobLevel','JobRole','JobSatisfaction','MaritalStatus','MonthlyIncome','MonthlyRate','NumCompaniesWorked','Over18','OverTime','PercentSalaryHike','PerformanceRating','RelationshipSatisfaction','StandardHours','StockOptionLevel','TotalWorkingYears','TrainingTimesLastYear','WorkLifeBalance','YearsAtCompany','YearsInCurrentRole','YearsSinceLastPromotion','YearsWithCurrManager'])
    
    # Ordinal encoding
    attrition_df[['BusinessTravel', 'Department', 'EducationField', 'Gender','JobRole', 'MaritalStatus', 'Over18', 'OverTime']]=label_attrition.transform(attrition_df[['BusinessTravel', 'Department', 'EducationField', 'Gender','JobRole', 'MaritalStatus', 'Over18', 'OverTime']])
    
    prediction = model_attrition.predict(attrition_df.values)

    if prediction == 1:
        output = "Employee is going to Churn. Further action required."
    else:
        output = "Employee is not going to Churn"

    return render_template('attrition.html', prediction_text=output)
    
@app.route('/attrition/info',methods=['POST'])
def attrition_info():
    return render_template('attrition_info.html')
    
@app.route('/loan',methods=['POST'])
def loan():
    return render_template('loan.html')
    
@app.route('/loan/predict',methods=['POST'])
def predict_loan():
    '''
    For rendering results on HTML GUI
    '''
    features = [x for x in request.form.values()]
    final_features_dummy = []
    
    # Drop features which are not important
    
    for i,v in enumerate(features):
        if (i == 1 or i == 5 or i == 7 or i == 8 or i == 9 or i == 11 or i == 12 or i == 14 or i == 15 or i == 16 or i == 17 or i == 18 or i == 19 or i == 20):
            final_features_dummy.append(v)
    
    # Need to convert features into float and string
    
    final_features = []

    for i,v in enumerate(final_features_dummy):
        if (i == 0 or i == 1 or i == 2 or i == 5 or i == 6 or i == 7):
            final_features.append(v)
        else:
    	    final_features.append(float(v))
    
    # Creating dataframe for ordinal encoding
    loan_df = pd.DataFrame([final_features], columns=['Gender','City_Category','Employer_Category1','Employer_Category2','Monthly_Income','Primary_Bank_Type','Contacted','Source_Category','Existing_EMI','Loan_Amount','Loan_Period','Interest_Rate','EMI','Var1'])
    
    # Ordinal encoding
    loan_df[['Gender','City_Category','Employer_Category1','Primary_Bank_Type','Contacted','Source_Category']]=label_loan.transform(loan_df[['Gender','City_Category','Employer_Category1','Primary_Bank_Type','Contacted','Source_Category']])
    
    prediction = model_loan.predict(loan_df.values)

    if prediction == 1:
        output = "Approved"
    else:
        output = "not Approved"    
    
    return render_template('loan.html', prediction_text='Loan {} for Customer Number : {}'.format(output,features[0]))
    
@app.route('/loan/info',methods=['POST'])
def loan_info():
    return render_template('loan_info.html')
    
@app.route('/house',methods=['POST'])
def house():
    return render_template('house.html')

@app.route('/house/predict',methods=['POST'])
def predict_house():
    '''
    For rendering results on HTML GUI
    '''
    int_features = [int(x) for x in request.form.values()]
    final_features = np.array([int_features])
    
    prediction = model_house.predict(final_features)[0]
        
    return render_template('house.html', prediction_text='House Price is {}'.format(round(prediction,2)))

@app.route('/sentiment',methods=['POST'])
def sentiment():
    return render_template('sentiment.html')

@app.route('/sentiment/predict',methods=['POST'])
def predict_sentiment():
    '''
    For rendering results on HTML GUI
    '''
    int_features = [x for x in request.form.values()]
    final_features = np.array(int_features)
    
    prediction = model_sentiment.predict(final_features)[0]

    if prediction == 1:
        output = "Positive Sentiment"
    else:
        output = "Negative Sentiment"
    
    return render_template('sentiment.html', prediction_text=output)
    
@app.route('/cluster',methods=['POST'])
def cluster():
    return render_template('cluster.html')

@app.route('/cluster/predict',methods=['POST'])
def predict_cluster():
    '''
    For rendering results on HTML GUI
    '''
    features = [x for x in request.form.values()]
    final_features = []
    
    # Need to convert features into numeric and string

    for i,v in enumerate(features):
        if (i == 1):
            pass
        else:
    	    final_features.append(int(v))
    
    final_features = np.array([final_features])
    
    prediction = model_cluster.predict(final_features[:,1:])

    if prediction == 0:
        output = "Customer belongs to Cluster 1 (Love Staples Items)"
    elif prediction == 1:
        output = "Customer belongs to Cluster 2 (Love Apparels)"
    else:
        output = "Customer belongs to Cluster 3 (Love FnV Items)"
    
    return render_template('cluster.html', prediction_text=output)
    
if __name__ == "__main__":
    app.run(debug=True)