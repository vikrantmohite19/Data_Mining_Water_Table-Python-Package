import pandas as pd
import numpy as np
from flask import Flask, request, render_template, jsonify
import os
import json
from src.pipeline.predict_pipeline import CustomData, PredictPipeline

app=Flask(__name__)


@app.route('/',methods=['GET'])
def home():
    return render_template('home.html') 


# @app.route('/predict_api',methods=['POST'])
# def predict_api():
#     data=request.json['data']
#     data = np.array(list(data.values())).reshape(1,-1)
  
#     Custom_Data = CustomData(data) 
#     df = Custom_Data.get_data_as_data_frame()

#     predict_pipeline=PredictPipeline()

#     y_pred, y_pred_proba = predict_pipeline.predict(df)  

#     arr_list1 = y_pred.tolist()  
#     json_str1 = json.dumps(arr_list1)  

#     arr_list2 = y_pred_proba.tolist()  
#     json_str2 = json.dumps(arr_list2)  
    
#     return jsonify(json_str1, json_str2)

@app.route('/predict',methods=['POST'])
def predict():


    data = [float(x) if isinstance(x, (int, float)) else x for x in request.form.values()]
    # data = list(request.form.values())
    data = (np.array(data)).reshape(1,-1)
    
    Custom_Data = CustomData(data) 
    df = Custom_Data.get_data_as_data_frame()


    predict_pipeline=PredictPipeline()

    y_pred, y_pred_proba = predict_pipeline.predict(df)
    y_pred = y_pred.tolist()
    y_pred_proba = y_pred_proba.tolist()
    return render_template("home.html",prediction_text="The status of waterwell is {} and the probabilities are{}".format(y_pred, y_pred_proba))




if __name__ == '__main__':
    # app.debug = True
    app.run(host='0.0.0.0')

