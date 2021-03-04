from flask import Flask, render_template, request
from sklearn.externals import joblib
import pandas as pd
import numpy as np

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('original.html')


@app.route("/predict", methods=['GET','POST'])
def predict():
    if request.method == 'POST':
        age = float(request.form['age'])
        sex = float(request.form['sex'])
        cp = float(request.form['cp'])
        trestbps = float(request.form['trestbps'])
        chol = float(request.form['chol'])
        fbs= float(request.form['fbs'])
        restecg = float(request.form['restecg'])
        thalach = float(request.form['thalach'])
        exang = float(request.form['exang'])
        oldpeak = float(request.form['oldpeak'])
        slope = float(request.form['slope'])
        ca = float(request.form['ca'])
        thal = float(request.form['thal'])

        pred_args = [age,sex,cp,trestbps,chol,fbs,restecg,thalach,exang,oldpeak,slope,ca,thal]

        mul_reg = open('heart_svm_13.pkl','rb')
        ml_model = joblib.load(mul_reg)
        model_predcition = ml_model.predict([pred_args])
        if model_predcition == 1:
            res = 'Affected'
        else:
            res = 'Not affected'
        #return res
    return render_template('predict.html', prediction = res)

if __name__ == '__main__':
    app.run()
