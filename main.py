'''
 OOPS Code for House Price Prediction
'''
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn
import sys
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score,root_mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from flask import Flask, render_template, request
import pickle,gzip
import warnings
warnings.filterwarnings("ignore")
app = Flask(__name__)
class Price_Prediction:
    def __init__(self,path):
        try:
            self.df=pd.read_csv(path)
            self.df=self.df.drop(['date','street'],axis=1)
            for col in ['city','statezip','country']:
                if col in self.df.columns:
                    self.df[col]=pd.factorize(self.df[col])[0]
            self.df = self.df[self.df['price'] > 0]
            self.df = self.df[self.df['price'] < self.df['price'].quantile(0.999)]
            self.X = self.df.iloc[:, 1:]
            self.y = self.df.iloc[:, 0]
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=42)
        except Exception as e:
            er_ty,er_msg,er_line=sys.exc_info()
            print(f"Error in line no.{er_line} due to {er_ty} reason is {er_msg}")

    def training(self):
        try:
            self.reg=RandomForestRegressor(n_estimators=200,max_depth=15)
            self.reg.fit(self.X_train,self.y_train)
            # Finding Training Accuracy and Loss
            self.train_predictions = self.reg.predict(self.X_train)
            print(f'the training accuracy is :{r2_score(self.y_train, self.train_predictions)}')
            print(f'the training loss is :{root_mean_squared_error(self.y_train, self.train_predictions)}')
        except Exception as e:
            er_ty,er_msg,er_line=sys.exc_info()
            print(f"Error in line no.{er_line} due to {er_ty} reason is {er_msg}")

    def testing(self):
        try:
            # Finding Testing Accuracy and Loss
            self.test_predictions = self.reg.predict(self.X_test)
            print(f'the testing accuracy is :{r2_score(self.y_test, self.test_predictions)}')
            print(f'the testing loss is :{root_mean_squared_error(self.y_test, self.test_predictions)}')
        except Exception as e:
            er_ty, er_msg, er_line = sys.exc_info()
            print(f"Error in line no.{er_line} due to {er_ty} reason is {er_msg}")

    def model_saving(self):
        try:
            with gzip.open("HPP_Model.pkl", 'wb') as f:
                pickle.dump(self.reg,f)
            #Load and Check
            with gzip.open("HPP_Model.pkl", 'rb') as p:
                m=pickle.load(p)
            result=m.predict([[4.0, 2.50, 1570, 1433, 3.0, 0, 0, 3, 1570, 0, 2010, 0, 1, 42, 0]])[0]
            print(f'Prediction value after loading the model:{result}')

        except Exception as e:
            er_ty, er_msg, er_line = sys.exc_info()
            print(f"Error in line no.{er_line} due to {er_ty} reason is {er_msg}")

if __name__=="__main__":
    try:
        dataset_path="data.csv"
        obj=Price_Prediction(dataset_path)
        obj.training()
        obj.testing()
        obj.model_saving()
    except Exception as e:
        er_ty, er_msg, er_line = sys.exc_info()
        print(f"Error in line no.{er_line} due to {er_ty} reason is {er_msg}")