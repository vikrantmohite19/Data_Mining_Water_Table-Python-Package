import sys
import pandas as pd
from src.exception import CustomException
from src.utils import load_object
import os
from src.logger import logging


class PredictPipeline:
    def __init__(self):
        pass

    def predict(self,df):
        try:

            scaler_path = os.path.join("preprocessed","scaler.pkl")
            encoder_path = os.path.join("preprocessed","encoder.pkl")
            model_path = os.path.join("artifacts","model.pkl")

            logging.info(f"Saving model path & vectoriser's path completed")

            scaler = load_object(file_path= scaler_path)
            encoder = load_object(file_path=encoder_path)
            model = load_object(file_path=model_path)
            
            logging.info(f"Laoding model object and vectoriser's object completed")

            numerical_features = ['gps_height','longitude', 'latitude', 'district_code','population', 'operational_year']
            categorical_features = ['funder','installer','basin', 'region', 'lga', 'extraction_type_group','management', 
                            'payment', 'water_quality', 'quantity', 'source','waterpoint_type']
            
            df[numerical_features] = scaler.transform(df[numerical_features])
            df[categorical_features] = encoder.transform(df[categorical_features])
            logging.info(f"Vectoriser's dataframe is ready")

            y_preds=model.predict(df)
            y_pred_proba = model.predict_proba(df)

            logging.info(f"prediction completed")
            return y_preds, y_pred_proba 
        
        except Exception as e:
            raise CustomException(e)



class CustomData:
    def __init__(self, df_list):
        self.df_list = df_list

    def get_data_as_data_frame(self):
        try:
            columns = ['funder','gps_height','installer','longitude', 'latitude',
                       'basin','region','district_code','lga','population',
                       'extraction_type_group','management','payment', 
                       'water_quality','quantity','source',
                       'waterpoint_type','operational_year']

            df = pd.DataFrame(self.df_list, columns=columns)
            logging.info(f"df is created")
            return df

        except Exception as e:
            raise CustomException(e)

