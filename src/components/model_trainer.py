import os
import sys
from dataclasses import dataclass

from catboost import CatBoostRegressor
from xgboost import XGBRegressor
from sklearn.ensemble import (AdaBoostRegressor, GradientBoostingRegressor, RandomForestRegressor)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor

from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, evaluate_model


@dataclass
class ModelTrainerConfig:
    trained_model_path = os.path.join("artifacts", "model.pkl")
    
class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()
        
    def initiate_model_trainer(self, train_arr, test_arr):
        try:
            logging.info("Split training and testing data")
            x_train, y_train, x_test, y_test = (
                train_arr[:, : -1],
                train_arr[:, -1], 
                test_arr[:, : -1],
                test_arr[:, -1]
            )
            models = {
                "Random Forest": RandomForestRegressor(), 
                "Decision Tree": DecisionTreeRegressor(), 
                "Gradient Boosting": GradientBoostingRegressor(), 
                "Linear Regression": LinearRegression(), 
                "KNeighbors Regressor": KNeighborsRegressor(), 
                "XGBRegressor": XGBRegressor(), 
                "CatBoost Regressor": CatBoostRegressor(), 
                "AdaBoostRegressor": AdaBoostRegressor()
            }
            
            model_report:dict = evaluate_model(x_train=x_train, y_train=y_train, 
                                               x_test=x_test, y_test=y_test, 
                                               models=models)
            
            # finding the best model, its score and getting its name from the dictionary
            best_model_score = max(sorted(model_report.values()))
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            best_model = models[best_model_name]
            
            if best_model_score < 0.65:
                raise CustomException("The best model does not exist.")
            
            logging.info(f"Best found model on both train and test set")

            save_object(
                file_path = self.model_trainer_config.trained_model_path,
                obj = best_model
            )

            best_model_pred = best_model.predict(x_test)

            r2_square = r2_score(y_test, best_model_pred)
            return r2_square
             
        except Exception as e:
            raise CustomException(e, sys)