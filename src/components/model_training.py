import sys
import os
from dataclasses import dataclass
from src.utils import evaluate_models

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
from src.utils import save_object

@dataclass
class ModelTrainerConfig:
    trained_model_file_path=os.path.join('artifacts','model.pkl')

class ModelTrainer:
    def __init__(self) -> None:
        self.model_trainer_config=ModelTrainerConfig()
    
    def initiate_model_trainer(self,train_arr,test_arr):
        try:
            logging.info("split train and test input data")
            x_train,y_train,x_test,y_test=(
                train_arr[:,:-1],
                train_arr[:,-1],
                test_arr[:,:-1],
                test_arr[:,-1]
            )
            models={
                "Random Forest":RandomForestRegressor(),
                "Decission Tree":DecisionTreeRegressor(),
                "Linear Regression":LinearRegression(),
                "Gradient Boosting":GradientBoostingRegressor(),
                "K-neighbors classifier":KNeighborsRegressor(),
                "XGBclassifier":XGBRegressor(),
                "Catboosting classifier":CatBoostRegressor(),
                "Adaboost classifier":AdaBoostRegressor()
            }

            model_report:dict=evaluate_models(x_train=x_train,y_train=y_train,x_test=x_test,
                                             y_test=y_test,models=models)
            best_model_score=max(sorted(model_report.values()))

            best_model_name=list(model_report.keys())[list(model_report.values()).index(best_model_score)]
            best_model=models[best_model_name]

            if best_model_score<0.6:
                raise CustomException("no best model found!")

            logging.info("best found model in training and testing datasets")
            save_object(
                file_path=ModelTrainerConfig.trained_model_file_path,
                obj=best_model
            )
            predicted = best_model.predict(x_test)
            r2_square=r2_score(y_test,predicted)
            return r2_square



        except Exception as e:
            raise CustomException(e,sys)


