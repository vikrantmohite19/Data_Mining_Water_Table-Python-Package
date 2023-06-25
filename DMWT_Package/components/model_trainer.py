import os
import numpy as np 
import pandas as pd
import sys
from dataclasses import dataclass
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_auc_score
from sklearn.metrics import f1_score

from DMWT_Package.exception import CustomException
from DMWT_Package.logger import logging

from DMWT_Package.utils import save_object,evaluate_models

@dataclass
class ModelTrainerConfig:
    trained_model_file_path=os.path.join("artifacts","model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()


    def initiate_model_trainer(self, X_train_smote_path,  y_train_smote_path, X_test_smote_path, y_test_smote_path):
        try:
            logging.info("reading vectorised data")
            X_train = pd.read_csv(X_train_smote_path)
            y_train = pd.read_csv(y_train_smote_path)
            X_test = pd.read_csv(X_test_smote_path)
            y_test = pd.read_csv(y_test_smote_path)
            
            logging.info("initializing the models and defining the parameters")
            
            
            models = {
		        "Logistic Regression": LogisticRegression(),
                "KNN": KNeighborsClassifier(),
                "Decision Tree": DecisionTreeClassifier(),
		        "Random Forest": RandomForestClassifier(),
    		    "XGBoost": XGBClassifier()

            }

            params = {

            "Logistic Regression": {
                'C': [0.1, 1.0, 10.0],  # Regularization parameter
                    # 'penalty': ['l1', 'l2'],  # Regularization type
                    'max_iter': [10, 20],  # Maximum number of iterations
                    # 'solver': ['liblinear', 'lbfgs', 'saga']  # Solver algorithm
                },


            "KNN": {
                    'n_neighbors': [3, 5, 7],  # Number of neighbors
                    # 'weights': ['uniform', 'distance'],  # Weight function used in prediction
                    # 'p': [1, 2]  # Power parameter for the Minkowski metric
                },


            "Decision Tree": {
			    'criterion': ['gini', 'entropy'],  # Splitting criterion
    			# 'max_depth': [None, 5, 10],  # Maximum depth of the tree
    			# 'min_samples_split': [2, 5, 10],  # Minimum number of samples required to split an internal node
    			# 'min_samples_leaf': [1, 2, 4],  # Minimum number of samples required to be at a leaf node                    
                },

            "Random Forest": {
			    'n_estimators': [10, 20, 30],  # Number of trees in the forest
    			# 'criterion': ['gini', 'entropy'],  # Splitting criterion
    			# 'max_depth': [5, 10],  # Maximum depth of the trees
    			# 'min_samples_split': [2, 5, 10],  # Minimum number of samples required to split an internal node
    			# 'min_samples_leaf': [1, 2, 4],  # Minimum number of samples required to be at a leaf node
		        },

		    "XGBoost": {
			    'max_depth': [3, 5, 7],  # Maximum depth of the trees
    			# 'learning_rate': [0.1, 0.01, 0.001],  # Learning rate
    			# 'n_estimators': [10, 15, 20],  # Number of trees
    			# 'gamma': [0, 0.1, 0.2],  # Minimum loss reduction required for a split
    			# 'subsample': [0.8, 1.0],  # Subsample ratio of the training instances
    			# 'colsample_bytree': [0.8, 1.0],  # Subsample ratio of columns when constructing each tree

                }
               
            }




            
            Result:dict=evaluate_models(X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test,
                                             models=models, param=params)
            
            
            ## To get best model score from dict
            best_model_score = max(sorted(Result['Test_Balanced_Accuracy']))

            ## To get best model name from dict

            best_model_name = Result['Model'][
                Result['Test_Balanced_Accuracy'].index(best_model_score)
            ]
            best_model = models[best_model_name]

            if best_model_score<0.6:
                raise CustomException("No best model found")
            logging.info(f"Best found model on both training and testing dataset")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )
            logging.info(f"Best model has been saved in artifact folder")
            
            predicted=best_model.predict(X_test)

            balanced_accuracy = balanced_accuracy_score(y_test, predicted)
            return balanced_accuracy, pd.DataFrame(Result)
            



            
        except Exception as e:
            raise CustomException(e)