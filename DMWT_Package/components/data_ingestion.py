import os
import sys
from DMWT_Package.exception import CustomException
from DMWT_Package.logger import logging
import pandas as pd

from sklearn.model_selection import train_test_split
from dataclasses import dataclass
from DMWT_Package.components.data_transformation import DataTransformation
from DMWT_Package.components.data_transformation import DataTransformationConfig

from DMWT_Package.components.model_trainer import ModelTrainerConfig
from DMWT_Package.components.model_trainer import ModelTrainer


@dataclass
class DataIngestionConfig:
    train_data_path: str=os.path.join('artifacts',"train.csv")
    test_data_path: str=os.path.join('artifacts',"test.csv")
    data_path: str=os.path.join('artifacts',"data.csv")


class DataIngestion:
    def __init__(self):
        self.ingestion_config=DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Entered the data ingestion method or component")
        try:
            df_data = pd.read_csv("notebook/data/train.csv")
            df_labels = pd.read_csv("notebook/data/train_lables.csv")
            df = df_data.merge(df_labels, how='left', on='id')
            logging.info('Read the dataset as dataframe')

            os.makedirs(os.path.dirname(self.ingestion_config.data_path),exist_ok=True)

            df.to_csv(self.ingestion_config.data_path,index=False,header=True)

            logging.info("Train test split initiated")
            train_set,test_set=train_test_split(df,test_size=0.2,random_state=42)

            train_set.to_csv(self.ingestion_config.train_data_path,index=False,header=True)

            test_set.to_csv(self.ingestion_config.test_data_path,index=False,header=True)

            logging.info("Inmgestion of the data is completed")

            return(
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path

            )
        except Exception as e:
            raise CustomException(e)

if __name__=="__main__":
    obj=DataIngestion()
    train_data,test_data=obj.initiate_data_ingestion()

    data_transformation=DataTransformation()
    preprocessed_train, preprocessed_test=data_transformation.save_preprocessed(train_data,test_data)

    X_smote_train_path, y_smote_train_path, X_smote_test_path, y_smote_test_path = data_transformation.vectorizer(preprocessed_train,preprocessed_test)

    
    modeltrainer=ModelTrainer()
    test_score, result_df = modeltrainer.initiate_model_trainer(X_smote_train_path, y_smote_train_path, X_smote_test_path, y_smote_test_path)
    print(test_score)
    print(result_df)
