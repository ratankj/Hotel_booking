

import os,sys
from hotel.exception import CustomException
from hotel.logger import logging
from hotel.constant import *
from hotel.config.configuration import PREPROCESSING_OBJ_PATH,TRANSFORMED_TRAIN_FILE_PATH,TRANSFORMED_TEST_FILE_PATH
from dataclasses import dataclass

from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler,PowerTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler,OrdinalEncoder
from sklearn.compose import ColumnTransformer
from category_encoders.binary import BinaryEncoder
from hotel.utils.utils import save_object

@dataclass
class DataTransformationConfig():
    preprocessor_obj_file_path=PREPROCESSING_OBJ_PATH
    transformed_train_path = TRANSFORMED_TRAIN_FILE_PATH
    transformed_test_path = TRANSFORMED_TEST_FILE_PATH


class DataTransformation():
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()


    def get_data_transformation_object(self):
        try:
            logging.info("Loading data transformation")
            
# binary
            
            binary_columns = ['country']
# numarical
            numerical_columns = ['hotel','lead_time','arrival_date_year','arrival_date_month',
                                 'arrival_date_day_of_month','stays_in_weekend_nights','stays_in_week_nights','previous_cancellations', 
                                 'reserved_room_type', 'booking_changes', 'adr', 'total_of_special_requests', 'total_guests']
# categorical features
            categorical_columns =['meal','market_segment','distribution_channel','deposit_type','customer_type']

            
            
            numerical_pipeline=Pipeline(steps=[
                ('impute',SimpleImputer()),
                ('scaler',StandardScaler()),
                ('transformer', PowerTransformer(method='yeo-johnson', standardize=False))
            ])

            binary_pipeline=Pipeline(steps=[
                ('imputer',SimpleImputer(strategy='most_frequent')),
                ('ordinal',BinaryEncoder()),
                ('scaler',StandardScaler())  

            ])

            categorical_pipeline=Pipeline(steps=[
                ('impute',SimpleImputer(strategy='most_frequent')),
                ('onehot',OneHotEncoder(handle_unknown='ignore')),
                ('scaler',StandardScaler(with_mean=False))
                ])

            preprocessor =ColumnTransformer([
                ('numerical_pipeline',numerical_pipeline,numerical_columns),
                ('binary_pipeline',binary_pipeline,binary_columns),
                ('category_pipeline',categorical_pipeline,categorical_columns)
            ])

            return preprocessor

            logging.info('pipeline completed')


        except Exception as e:
            logging.info("Error getting data transformation object")
            raise CustomException(e,sys)
        


    


    def _remove_outliers_IQR(self, col, df):
        try:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            iqr = Q3 - Q1
            upper_limit = Q3 + 1.5 * iqr
            lower_limit = Q1 - 1.5 * iqr
            df.loc[(df[col]>upper_limit), col]= upper_limit
            df.loc[(df[col]<lower_limit), col]= lower_limit 
            return df
        
        except Exception as e:
            logging.info(" outlier code")
            raise CustomException(e, sys) from e 
        

    
        






    
    def initaite_data_transformation(self,train_path,test_path):
        try:
            # Reading train and test data
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info('Read train and test data completed')
            logging.info(f'Train Dataframe Head : \n{train_df.head().to_string()}')
            logging.info(f'Test Dataframe Head  : \n{test_df.head().to_string()}')


            logging.info(f"columns in dataframe are: {train_df.columns}")

            logging.info(f"columns in dataframe are: {train_df.dtypes}")



            # transforming child, adult and babies into total guest column
            train_df["total_guests"]= train_df["children"]+train_df["adults"]+train_df["babies"] 
            test_df["total_guests"]= test_df["children"]+test_df["adults"]+test_df["babies"] 



            # transforming hotel,arrival_date_month

            train_df['hotel'] = train_df['hotel'].map({'Resort Hotel':0, 'City Hotel':1})
            train_df['arrival_date_month'] = train_df['arrival_date_month'].map({'January':1, 'February': 2, 'March':3, 'April':4, 'May':5, 'June':6, 'July':7,
                                                            'August':8, 'September':9, 'October':10, 'November':11, 'December':12})
            train_df['reserved_room_type'] = train_df['reserved_room_type'].map({'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6, 'H': 7, 'L': 8})



            test_df['hotel'] = test_df['hotel'].map({'Resort Hotel':0, 'City Hotel':1})
            test_df['arrival_date_month'] = test_df['arrival_date_month'].map({'January':1, 'February': 2, 'March':3, 'April':4, 'May':5, 'June':6, 'July':7,
                                                            'August':8, 'September':9, 'October':10, 'November':11, 'December':12})
            test_df['reserved_room_type'] = test_df['reserved_room_type'].map({'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6, 'H': 7, 'L': 8})
            
            
            # tranforming lead time

            train_df["lead_time"]= (train_df["lead_time"]/24).round(2)
            test_df["lead_time"]= (test_df["lead_time"]/24).round(2)


            
            # replace meal Undefined with Self Catering 

            train_df["meal"].replace("Undefined", "SC", inplace=True)
            # Replace 
            train_df["market_segment"].replace("Undefined", "Online TA", inplace=True)
            train_df.drop(train_df[train_df['distribution_channel'] == 'Undefined'].index, inplace=True, axis=0)

            
            test_df["meal"].replace("Undefined", "SC", inplace=True)
            # Replace 
            test_df["market_segment"].replace("Undefined", "Online TA", inplace=True)
            test_df.drop(test_df[test_df['distribution_channel'] == 'Undefined'].index, inplace=True, axis=0)




            # drop columns
            
            train_df.drop(columns=['arrival_date_week_number','reservation_status','reservation_status_date',
                 'assigned_room_type','agent','required_car_parking_spaces', 'is_repeated_guest',
                 'previous_bookings_not_canceled','days_in_waiting_list',
                 'company', 'name','email','phone-number','credit_card','babies','adults','children'], inplace=True, axis=1)



            logging.info(f"columns in dataframe are: {train_df.columns}")


            test_df.drop(columns=['arrival_date_week_number','reservation_status','reservation_status_date',
                 'assigned_room_type','agent','required_car_parking_spaces', 'is_repeated_guest',
                 'previous_bookings_not_canceled','days_in_waiting_list',
                 'company', 'name','email','phone-number','credit_card','babies','adults','children'], inplace=True, axis=1)
            

            logging.info(f"columns in dataframe are: {test_df.columns}")


# ['hotel', 'lead_time', 'arrival_date_year', 'arrival_date_month', 'arrival_date_day_of_month',
#  'stays_in_weekend_nights', 'stays_in_week_nights', 'meal', 'country', 'market_segment', 'distribution_channel',
#  'previous_cancellations', 'reserved_room_type', 'booking_changes', 'deposit_type', 'customer_type', 'adr', 
# 'total_of_special_requests', 'total_guests']

#meal 
#country :177
#market_segment 
#distribution_channel 
#deposit_type 
#customer_type 
            
           

            # Assuming 'df' is your DataFrame
            num_col = [feature for feature in train_df.columns if train_df[feature].dtype != '0']
            
            logging.info(f"numerical_columns: {num_col}")


            cat_col = [feature for feature in train_df.columns if train_df[feature].dtype == 'O']
            logging.info(f"numerical_columns: {cat_col}")


            train_df['arrival_date_month'] = train_df['arrival_date_month'] .astype(int)
            test_df['arrival_date_month'] = test_df['arrival_date_month'] .astype(int)

            numerical_columns = ['hotel','lead_time','arrival_date_year','arrival_date_month',
                                 'arrival_date_day_of_month','stays_in_weekend_nights','stays_in_week_nights','previous_cancellations', 
                                 'reserved_room_type', 'booking_changes', 'adr', 'total_of_special_requests', 'total_guests']

            


            for col in numerical_columns:
                self._remove_outliers_IQR(col=col, df= train_df)

            
            for col in numerical_columns:
                self._remove_outliers_IQR(col=col, df= test_df)
                
            logging.info(f"Outlier capped in test and train df") 



            preprocessing_obj = self.get_data_transformation_object()

            logging.info(f"Train Dataframe Head:\n{train_df.head().to_string()}")
            logging.info(f"Test Dataframe Head:\n{test_df.head().to_string()}")



            target_column_name = 'is_canceled'



            X_train = train_df.drop(columns=target_column_name,axis=1)
            y_train = train_df[target_column_name]

            X_test = test_df.drop(columns=target_column_name,axis=1)
            y_test = test_df[target_column_name]


            logging.info(f"shape of {X_train.shape} and {y_train.shape}")
            logging.info(f"shape of {X_test.shape} and {y_test.shape}")

            # Transforming using preprocessor obj
            
            X_train=preprocessing_obj.fit_transform(X_train)            
            X_test=preprocessing_obj.transform(X_test)

            logging.info("Applying preprocessing object on training and testing datasets.")
            logging.info(f"shape of {X_train.shape} and {y_train.shape}")
            logging.info(f"shape of {X_test.shape} and {y_test.shape}")
            


            logging.info("transformation completed")



            train_arr =np.c_[X_train,np.array(y_train)]
            test_arr =np.c_[X_test,np.array(y_test)]
            

            logging.info("train arr , test arr")


            df_train= pd.DataFrame(train_arr)
            df_test = pd.DataFrame(test_arr)

            logging.info("converting train_arr and test_arr to dataframe")
            logging.info(f"Final Train Transformed Dataframe Head:\n{df_train.head().to_string()}")
            logging.info(f"Final Test transformed Dataframe Head:\n{df_test.head().to_string()}")

            os.makedirs(os.path.dirname(self.data_transformation_config.transformed_train_path),exist_ok=True)
            df_train.to_csv(self.data_transformation_config.transformed_train_path,index=False,header=True)

            logging.info("transformed_train_path")
            logging.info(f"transformed dataset columns : {df_train.columns}")

            os.makedirs(os.path.dirname(self.data_transformation_config.transformed_test_path),exist_ok=True)
            df_test.to_csv(self.data_transformation_config.transformed_test_path,index=False,header=True)

            logging.info("transformed_test_path")

            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj)
            
            logging.info("Preprocessor file saved")
            
            return(train_arr,
                   test_arr,
                   self.data_transformation_config.preprocessor_obj_file_path)

        except Exception as e:
            raise CustomException(e, sys) from e 

