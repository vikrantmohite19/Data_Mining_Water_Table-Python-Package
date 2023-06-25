import sys
from dataclasses import dataclass
import numpy as np 
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from DMWT_Package.exception import CustomException
from DMWT_Package.logger import logging
import os
from DMWT_Package.utils import save_object
from category_encoders import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
from category_encoders import TargetEncoder, LeaveOneOutEncoder, WOEEncoder
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.model_selection import train_test_split
import category_encoders as ce
from imblearn.over_sampling import SMOTE



@dataclass
class DataTransformationConfig:
    train_processed_path: str=os.path.join('preprocessed',"train_preprocessed.csv")
    test_processed_path: str=os.path.join('preprocessed',"test_preprocessed.csv")


    train_vectorized_path: str=os.path.join('preprocessed',"train_vectorized.csv")
    test_vectorized_path: str=os.path.join('preprocessed',"test_vectorized.csv")

    scaler_obj_path=os.path.join('preprocessed',"scaler.pkl")
    encoder_obj_path=os.path.join('preprocessed',"encoder.pkl")

    X_train_smote_path=os.path.join('preprocessed',"X_train_smote.csv")
    y_train_smote_path=os.path.join('preprocessed',"y_train_smote.csv")
    X_test_smote_path=os.path.join('preprocessed',"X_test_smote.csv")
    y_test_smote_path=os.path.join('preprocessed',"y_test_smote.csv")




class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()

    def data_preprocessor(self, file_path):
        '''
        This function si responsible for data trnasformation
        
        '''
        try:
            df=pd.read_csv(file_path)

            df['funder'].fillna(value='Undefined',inplace=True) 
            df['funder'].replace(to_replace = '0', value ='Undefined' , inplace=True) #replacing '0' & missing values with 'Undefined'
            top_30_funders = ['Government Of Tanzania','Undefined','Danida','Hesawa','Rwssp','World Bank','Kkkt','World Vision',
                      'Unicef','Tasaf','District Council','Dhv','Private Individual','Dwsp','Norad','Germany Republi',
                      'Tcrs','Ministry Of Water','Water','Dwe','Netherlands','Hifab','Adb','Lga','Amref','Fini Water',
                      'Oxfam','Wateraid','Rc Church','Isf']
            df.loc[~df["funder"].isin(top_30_funders), "funder"] = "other"
            
            df['installer'].fillna(value='Undefined',inplace=True) 
            df['installer'].replace(to_replace = '0', value ='Undefined' , inplace=True) #replacing '0' category with 'Undefined'


            df['installer'].replace(to_replace = ("Gove","Gover","GOVERM", "GOVERN", "GOVERNME", "Governmen","Government",
                                                "GOVER"), value ="Government", inplace=True)
            df['installer'].replace(to_replace = ("RW","RWE","RWE /Community","RWE Community","RWE/ Community","RWE/Community",
                                                "RWE/DWE","RWE/TCRS","RWEDWE","RWET/WESA"), value ="RWE", inplace=True)
            df['installer'].replace(to_replace = ("Commu", "Communit", "Community", "COMMUNITY BANK",
                                                "Comunity"), value ="Community", inplace=True)
            df['installer'].replace(to_replace = ("Danda","DANIAD","Danid","DANIDA","DANIDA CO","DANIDS","DANNIDA",
                                                "DANID"), value ="DANIDA", inplace=True)
            df['installer'].replace(to_replace = ("Cebtral Government","Cental Government","Centr","Centra Government","Centra govt",
                                                "Central government","Central govt","Cetral government /RC","Tanzania Government",
                                                "TANZANIAN GOVERNMENT"), value ="Central Government", inplace=True)
            df['installer'].replace(to_replace = ("COUN","Counc","Council","Distri","District  Council",
                                                "District Community j","District Counci", "District council",
                                                "District Council"), value ="District Council", inplace=True)
            df['installer'].replace(to_replace = ("Hesawa","HESAW","Hesewa","HESAWA"),value ='HESAWA' , inplace=True)
            df['installer'].replace(to_replace = ("World Division","World Visiin","World vision","World Vission",
                                                "World Vision"),value ='World Vision' , inplace=True)
            df['installer'].replace(to_replace = ("Distric Water Department","District Water Department",
                                                "District water depar","District water department",
                                                "Water Department"),value ='District water department' , inplace=True)
            df['installer'].replace(to_replace = ("FINN WATER","FinW","FinWate","FinWater","Fini water",
                                                "Fini Water" ),value ='Fini Water' , inplace=True)
            df['installer'].replace(to_replace = ("RC","RC .Church","RC C","RC Ch","RC Churc","RC Church",
                                                "RC CHURCH BROTHER","RC church/CEFA","RC church/Central Gover",
                                                "RCchurch/CEFA","RC CHURCH"),value ="RC Church" , inplace=True)
            df['installer'].replace(to_replace = ("Villa","VILLAGER","Villagerd","Villagers","Villages","Villege Council",
                                                "Villi","villigers"),value ="Villagers" , inplace=True)


            top_30_installer = ["DWE","Undefined","Government","DANIDA","Community","HESAWA","RWE","District Council",
                                "Central Government","KKKT","TCRS","World Vision","CES","Fini Water","RC Church","LGA",
                                "WEDECO","TASAF","AMREF","TWESA","WU","Dmdd","ACRA","Villagers","SEMA","DW","OXFAM","Da",
                                "Idara ya maji","UNICEF"]


            df.loc[~df["installer"].isin(top_30_installer), "installer"] = "other"
            
            df['longitude'].replace(to_replace = 0 , value =34.07742669202832 , inplace=True)
            df['population'].replace(to_replace = 0, value = 281.087167 , inplace=True)
            df['construction_year'].replace(to_replace = 0, value = 1996, inplace=True)
            df['date_recorded'] = pd.to_datetime(df['date_recorded']) #converting dates to 'datetime' datatype 
            df['operational_year'] = df.date_recorded.dt.year - df.construction_year
            df.operational_year.head(5)
            df.loc[df['operational_year']<0, 'operational_year'] = 0
            
            df.drop(columns=["construction_year", "date_recorded","extraction_type", "extraction_type_class",'payment_type',
                        "quality_group", "quantity_group","source_type", "source_class","waterpoint_type_group",'permit',
                        "scheme_management", 'id', 'amount_tsh','ward','wpt_name', 'num_private','subvillage','region_code',
                        'public_meeting','recorded_by','management_group','scheme_name'], inplace=True)
            
            

            return df
        
        except Exception as e:
            raise CustomException(e)
        


    def save_preprocessed(self, train_df_path, test_df_path):

        try:
    
            pre_train_df = self.data_preprocessor(train_df_path)
            pre_test_df = self.data_preprocessor(test_df_path)

            logging.info(f"preprocessing completes.")

            os.makedirs(os.path.dirname(self.data_transformation_config.train_processed_path),exist_ok=True)

            pre_train_df.to_csv(self.data_transformation_config.train_processed_path,index=False,header=True)
            pre_test_df.to_csv(self.data_transformation_config.test_processed_path,index=False,header=True)

            logging.info(f"preprocessed csv files are saved in 'preprocessed folder.")

            return  self.data_transformation_config.train_processed_path, self.data_transformation_config.test_processed_path
            

        except Exception as e:
            raise CustomException(e)


    def vectorizer(self, train_processed_path, test_processed_path):


        try: 
            df_train = pd.read_csv(train_processed_path)
            df_test = pd.read_csv(test_processed_path)

            logging.info(f"reading the prrprocessed data is done.")

            numeric_target_values = {'functional':1, 'non functional':0, 'functional needs repair':2}

            df_train['status_group'] = df_train['status_group'].replace(numeric_target_values)
            df_test['status_group'] = df_test['status_group'].replace(numeric_target_values)


            #encoding target variables manualy 

            numerical_features = ['gps_height','longitude', 'latitude', 'district_code','population', 'operational_year']
            categorical_features = ['funder','installer','basin', 'region', 'lga', 'extraction_type_group','management', 
                                    'payment', 'water_quality', 'quantity', 'source','waterpoint_type']
            
            y_train=df_train['status_group']
            y_test=df_test['status_group']

            X_train = df_train.drop(columns = ['status_group'])
            X_test = df_test.drop(columns = ['status_group'])

            
            scaler = RobustScaler()
            encoder = ce.TargetEncoder()

            scaler = scaler.fit(X_train[numerical_features])


            save_object(

                file_path=self.data_transformation_config.scaler_obj_path,
                obj=scaler
            )

            logging.info(f"saving the scaler object is done.")

            X_train[numerical_features] = scaler.transform(X_train[numerical_features])
            X_test[numerical_features] = scaler.transform(X_test[numerical_features])

            encoder = encoder.fit(X_train[categorical_features], y_train)

            save_object(

                file_path=self.data_transformation_config.encoder_obj_path,
                obj=encoder
            )

            logging.info(f"saving the encoder object is done.")

            X_train[categorical_features] = encoder.transform(X_train[categorical_features])
            X_test[categorical_features] = encoder.transform(X_test[categorical_features])

            logging.info(f"applying SMOTE on train data.")
            smote1 = SMOTE(sampling_strategy = 'auto', n_jobs = -1)
            X_smote_train, y_smote_train = smote1.fit_resample(X_train, y_train)
            y_smote_train = pd.Series(y_smote_train)

            logging.info(f"applying SMOTE on test data.")
            smote2 = SMOTE(sampling_strategy = 'auto', n_jobs = -1)
            X_smote_test, y_smote_test = smote2.fit_resample(X_test, y_test)
            y_smote_test = pd.Series(y_smote_test)
            
            X_smote_train.to_csv(self.data_transformation_config.X_train_smote_path,index=False,header=True)
            y_smote_train.to_csv(self.data_transformation_config.y_train_smote_path,index=False,header=True)
            X_smote_test.to_csv(self.data_transformation_config.X_test_smote_path,index=False,header=True)
            y_smote_test.to_csv(self.data_transformation_config.y_test_smote_path,index=False,header=True)

            logging.info(f"saving all CSV files after applying SMOTE")

            return (
                self.data_transformation_config.X_train_smote_path, 
                self.data_transformation_config.y_train_smote_path, 
                self.data_transformation_config.X_test_smote_path, 
                self.data_transformation_config.y_test_smote_path
            )    
            
        

        except Exception as e:
                raise CustomException(e)
        


