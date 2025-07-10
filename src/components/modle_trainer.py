import os 
import sys 
from dataclasses import dataclass 
from catboost import CatBoostRegressor
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor
)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from src.exception import CustomException
from src.logger import logging 
from src.utils import save_object ,evaluate_model

@dataclass
class modeltrainerconfig:
    trained_model_file_path = os.path.join('artifacts',"model.pkl")

class Modeltrainer:
    def __init__(self):
        self.model_trainer_config = modeltrainerconfig()


    def initiate_model_trainer(self,train_array,test_array):
        try:
            logging.info("Split training and test input") 
            X_train,y_train,X_test,y_test = (
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            ) 

            models = {
                   "Linear Regression": LinearRegression(),
                   "K-Neighbors Regressor": KNeighborsRegressor(),
                   "XGBoost Regressor": XGBRegressor(),
                   "Catboost Regressor": CatBoostRegressor(verbose=False),
                   "Random Forest Regressor": RandomForestRegressor(),
                   "AdaBoost Regressor": AdaBoostRegressor(),
                   "Decision Tree Regressor": DecisionTreeRegressor(),
                   "Gradient Boosting": GradientBoostingRegressor()
            }
            params = {
                "Decision Tree Regressor":{
                    'criterion':['squared_error','friedman_mse','absolute_error','poisson'],
                },
                "Random Forest Regressor":{
                    'n_estimators':[8,16,32,64,128,256]
                },
                "Gradient Boosting":{
                    'learning_rate':[.1,.01,.05,.001],
                    'subsample':[0.6,0.7,0.75,0.8,0.85,0.9],
                    'n_estimators':[8,16,32,64,128,256]
                },
                "Linear Regression":{},
                "K-Neighbors Regressor":{
                    'n_neighbors':[5,7,9,11],
                },
                "XGBoost Regressor":{
                    'learning_rate':[.1,.01,.05,.001],
                    'n_estimators':[8,16,32,64,128,256]
                },
                "Catboost Regressor":{
                    'depth':[6,8,10],
                    'learning_rate':[0.01,0.05,0.1],
                    'iterations':[30,50,100]
                },
                "AdaBoost Regressor":{
                     'learning_rate':[.1,.01,.05,.001],
                    'n_estimators':[8,16,32,64,128,256]
                },
            }

            model_report:dict=evaluate_model(X_train,y_train,X_test,y_test,models=models,param = params)

            # to get best model score from dict
            best_model_score = max(sorted(model_report.values()))

            # to get best model name from dict 
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]

            best_model = models[best_model_name]
            
            if best_model_score < 0.6:
                raise CustomException("No best model found")
            logging.info("Best found model on both training and testing dataset")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj = best_model
            )

            predicted = best_model.predict(X_test)

            r2_scores = r2_score(y_test,predicted)
            return r2_scores

        except Exception as e:
            raise CustomException(e,sys)         