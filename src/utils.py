import os
import sys

import numpy as np 
import pandas as pd
import dill
import pickle
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_auc_score
from sklearn.metrics import f1_score
from src.exception import CustomException

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)
    


def load_object(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj)

    except Exception as e:
        raise CustomException(e, sys)


def evaluate_models(X_train, y_train,X_test,y_test,models,param):
    try:
        Result = {}

        Result = {
            'Model': [],
            'Train_Accuracy_score' : [],
            'Train_Accuracy_score': [],
            'Train_Balanced_Accuracy': [],
            'Test_Balanced_Accuracy' : [],
            'Train_F1_Score': [],
            'Test_F1_Score': [],
            'Train_roc_auc_score': [],
            'Test_roc_auc_score': []
        }

        for i in range(len(list(models))):
            model = list(models.values())[i]
            para=param[list(models.keys())[i]]

            gs = GridSearchCV(model,para,cv=3)
            gs.fit(X_train,y_train)

            model.set_params(**gs.best_params_)
            model.fit(X_train,y_train)


            y_train_pred = model.predict(X_train)
            y_train_pred_proba = model.predict_proba(X_train)
            y_test_pred = model.predict(X_test)
            y_test_pred_proba = model.predict_proba(X_test)

            Train_Accuracy_score = accuracy_score(y_train, y_train_pred)
            Test_Accuracy_score = accuracy_score(y_test, y_test_pred)

            Result['Model'].append(list(models.keys())[i])
            Result['Train_Accuracy_score'].append(Train_Accuracy_score)
            Result['Test_Accuracy_score'].append(Test_Accuracy_score)


            Train_Balanced_Accuracy = balanced_accuracy_score(y_train, y_train_pred)
            Test_Balanced_Accuracy = balanced_accuracy_score(y_test, y_test_pred)

            
            Result['Train_Balanced_Accuracy'].append(Train_Balanced_Accuracy)
            Result['Test_Balanced_Accuracy'].append(Test_Balanced_Accuracy)

            Train_F1_Score = f1_score(y_train, y_train_pred, average="weighted")
            Test_F1_Score = f1_score(y_test, y_test_pred, average="weighted")

            Result['Train_F1_Score'].append(Train_F1_Score)
            Result['Test_F1_Score'].append(Test_F1_Score)

            Train_roc_auc_score = roc_auc_score(y_train, y_train_pred_proba, multi_class='ovr')
            Test_roc_auc_score = roc_auc_score(y_test, y_test_pred_proba, multi_class='ovr')


            Result['Train_roc_auc_score'].append(Train_roc_auc_score)
            Result['Test_F1_Score'].append(Test_roc_auc_score)

        
        return Result

    except Exception as e:
        raise CustomException(e)
