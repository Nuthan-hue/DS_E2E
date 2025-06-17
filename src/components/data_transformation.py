import os
import sys
from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer # To replace missing values with mean, median, mode..
from src.utils import save_object

from src.logger import logging
from src.exception import CustomException


@dataclass
class DataTransformationConfig:# To get input data/file_path that I require for data transformation i have created  DataTransformationConfig
    preprocessor_obj_file_path= os.path.join("artifact",'preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.datatransformation_config= DataTransformationConfig()

    def get_data_transformer_object(self):
        '''
        This function is responsible for data transformation
        '''
        try:
            numerical_columns=['reading_score', 'writing_score']
            categorical_columns=['gender', 'race_ethnicity', 'parental_level_of_education', 'lunch', 'test_preparation_course']
            numerical_pipeline=Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")), #To handle missing values
                    ("scalar",StandardScaler())
                ]
            )
            categorical_pipeline=Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("one_hot_encoder",OneHotEncoder()),
                    ("scalar", StandardScaler())
                ]
            )
            logging.info("categorical columns encoding completed")
            logging.info("Numerical columns standard scaling completed")
            preprocessor= ColumnTransformer(
                [
                    ("numerical_pipeline",numerical_pipeline, numerical_columns),
                    ("categorical_pipeline",categorical_pipeline,categorical_columns)
                ]
            )

            return preprocessor
        except Exception as e:
            raise CustomException(e, sys)

    def data_initiation(self, train_path, test_path):
        try:
            train_df=pd.read_csv("train_path")
            test_df=pd.read_csv("test_path")
            logging.info("Read Train and Test completed")
            logging.info("Obtaining preprocessingg object")
            preprocessing_obj=self.get_data_transformer_object()
            target_column_name= "math_score"
            numerical_columns=["writing_score","reading_score"]
            input_feature_train_df=train_df.drop(columns=[target_column_name], axis=1)
            target_feature_train_df=train_df[target_column_name]
            input_feature_test_df=train_df.drop(columns=[target_column_name], axis=1)
            target_feature_test_df=train_df[target_column_name]
            logging.info("Applying preprocessing bjject on training data frame and testing dataframe")
            input_feature_train_arr=preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr=preprocessing_obj.transform(input_feature_test_df)
            train_arr=np.c_[
                input_feature_train_arr,np.array(target_feature_train_df)
            ]
            test_arr=np.c_[
                input_feature_test_arr,np.array(target_feature_test_df)
            ]
            logging.info("saved preprocessing object.")
            save_object(#save pikel file
                file_path=self.datatransformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )
            return(
                train_arr,
                test_arr,
                self.datatransformation_config.preprocessor_obj_file_path
            )
        except:
            pass