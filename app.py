import pandas as pd
import numpy as np
from flask import Flask, request, render_template, jsonify
import os
import json
import pickle


app=Flask(__name__)

scaler = pickle.load(open('scaler.pkl','rb'))
encoder = pickle.load(open('encoder.pkl','rb'))
model=pickle.load(open('model.pkl','rb'))   


columns = ['funder','gps_height','installer','longitude', 'latitude','basin','region','district_code','lga','population','extraction_type_group','management','payment', 'water_quality','quantity','source','waterpoint_type','operational_year']


def DMWT_prediction1(df):
    numerical_features = ['gps_height','longitude', 'latitude', 'district_code','population', 'operational_year']
    categorical_features = ['funder','installer','basin', 'region', 'lga', 'extraction_type_group','management', 
                            'payment', 'water_quality', 'quantity', 'source','waterpoint_type']
    
    scaler = pickle.load(open('scaler.pkl','rb'))
    encoder = pickle.load(open('encoder.pkl','rb'))
    
    df[numerical_features] = scaler.transform(df[numerical_features])
    df[categorical_features] = encoder.transform(df[categorical_features])

    model=pickle.load(open('model.pkl','rb'))
    y_pred = model.predict(df)
    y_pred_proba = model.predict_proba(df)
    
    return y_pred, y_pred_proba 


@app.route('/',methods=['GET'])
def home():
    return render_template('home.html') 


@app.route('/predict_api',methods=['POST'])
def predict_api():
    data=request.json['data']
    data = np.array(list(data.values())).reshape(1,-1)
    data = pd.DataFrame(data, columns=columns) 

    y_pred, y_pred_proba = DMWT_prediction1(data)

    arr_list1 = y_pred.tolist()  
    json_str1 = json.dumps(arr_list1)  

    arr_list2 = y_pred_proba.tolist()  
    json_str2 = json.dumps(arr_list2)  


    return jsonify(json_str1, json_str2)

@app.route('/predict',methods=['POST'])
def predict():
    data = [float(x) if isinstance(x, (int, float)) else x for x in request.form.values()]
    data = list(request.form.values())
    data = (np.array(data)).reshape(1,-1)
    data = pd.DataFrame(data, columns=columns) 

    y_pred, y_pred_proba = DMWT_prediction1(data)

    y_pred = y_pred.tolist()  
    # json_str1 = json.dumps(arr_list1)  

    y_pred_proba = y_pred_proba.tolist()  
    # json_str2 = json.dumps(arr_list2)  


    # Output =  jsonify(json_str1, json_str2)

    return render_template("home.html",prediction_text="The status of waterwell is {} and the probabilities are{}".format(y_pred, y_pred_proba))
    # return render_template("home.html", prediction_text=jsonify({json_str1, json_str2}))



if __name__ == '__main__':
    app.debug = True
    app.run(host='0.0.0.0', port=9900)

