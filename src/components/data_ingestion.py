import os
import sys
from src.exception import CustomException
from src.logger import logging
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass

@dataclass #Inside a class to define a class variable we use __init__. With this decorator npo need of __init__
class DataIngestionConfiguration:
    train_data_path: str = os.path.join('artifact','train.csv')
    test_data_path: str = os.path.join('artifact','test.csv')
    raw_data_path: str = os.path.join('artifact','data.csv')

class DataIngestion:#here I have other functions inside the class, not just variables just like DataIngestionConfiguration. So I write __init__
    def __init__(self):
        self.ingestionConfig=DataIngestionConfiguration
    
    def initiate_data_injection(self):
        logging.info("Started initiate_data_injection function to read csv data to a dataframe")
        try:
            df=pd.read_csv("notebook/data/stud.csv")
            logging.info("Read the data to dataframe")
            os.makedirs(os.path.dirname(self.ingestionConfig.train_data_path),exist_ok=True)# created a directory as per ingestionConfig
            df.to_csv(self.ingestionConfig.raw_data_path, index=False, header= True)# save the df into raw_data_path in artifact folder
            logging.info("Train test split initiated")
            train_set, test_set= train_test_split(df, test_size=0.2, random_state=23)
            train_set.to_csv(self.ingestionConfig.train_data_path, index= False, header= True) #save the train dataset into train.csv in artifact folder
            test_set.to_csv(self.ingestionConfig.test_data_path, index= False, header= True) #save the train dataset into test.csv in artifact folder
            logging.info("Injection of the data is completed")
            return(
                self.ingestionConfig.test_data_path,
                self.ingestionConfig.train_data_path
            )
        except Exception as e:
            raise CustomException(e,sys)

if __name__=='__main__':
    object=DataIngestion()
    object.initiate_data_injection()
