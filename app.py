import flask
from flask import Flask,render_template,request
import numpy as np
import pandas as pd
import pickle
import gzip
import sklearn
from sklearn.ensemble import RandomForestRegressor

with gzip.open("HPP_Model.pkl",'rb') as f:
    model=pickle.load(f)

df=pd.read_csv("data.csv")
city_list=list(pd.factorize(df.iloc[:,-3])[1])
statezip_list=list(pd.factorize(df.iloc[:,-2])[1])
app=Flask(__name__)
@app.route("/")
def fun():
    return render_template("index.html")
@app.route("/predict",methods=['GET','POST'])
def prediction():
    raw_values=list(request.form.values())
    values=[]
    for i,val in enumerate(raw_values):
        if i==len(raw_values)-3:
             if val in city_list:
                 values.append(float(city_list.index(val)))
             else:
                 values.append(float(val))
        elif i==len(raw_values)-2:
             if val in statezip_list:
                 values.append(float(statezip_list.index(val)))
             else:
                 values.append(float(val))
        else:
            values.append(0.0)

    sol=model.predict([values])[0]
    return render_template("index.html",prediction_text=sol)


if __name__=="__main__":
    app.run(debug=True)