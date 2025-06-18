import os
import sys
import pandas as pd
import numpy as np
import dill
from sklearn.metrics import r2_score
from src.exception import CustomException
from src.logger import logging
from sklearn.model_selection import GridSearchCV


def save_object(file_path,obj):
    try:
        dir_path=os.path.dirname(file_path)
        os.makedirs(dir_path,exist_ok=True)
        with open(file_path,"wb") as file_obj:
            dill.dump(obj,file_obj)
        logging.info("Try bloack did well")
        #print("Try block did well")
    except Exception as e:
        raise CustomException(e,sys)

def evaluate_model(X_train, X_test, y_train, y_test, models, parameters):
    try:
        report = {}
        for i in range(len(list(models))):
            model = list(models.values())[i]
            para=parameters[list(models.keys())[i]]
            gs = GridSearchCV(model,para,cv=3)
            gs.fit(X_train,y_train)
            print(gs.best_params_)  # best tuned params
            print(gs.best_estimator_.get_params())  # all params of the best model
            model.set_params(**gs.best_params_)
            model.fit(X_train,y_train)
            model_name = list(models.keys())[i]

            #model.fit(X_train, y_train)
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)
            train_model_score = r2_score(y_train, y_train_pred)
            test_model_score = r2_score(y_test, y_test_pred)
            report[model_name] = test_model_score
        return report

    except Exception as e:
        raise Exception(e,sys)


