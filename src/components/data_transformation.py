import sys
import os
from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object


@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join("artifacts", "preprocessr.pkl")
    
class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()
        
    def get_data_transformer_obj(self): # to create all the pkl files to convert feature scaling etc.
        try:
            numerical_features = ['reading_score', 'writing_score']
            categorical_features = [
                "gender",
                "race_ethnicity",
                "parental_level_of_education",
                "lunch",
                "test_preparation_course"
            ]
            
            numerical_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy='median')),
                    ("scaler", StandardScaler(with_mean=False))
                ]
            )
            
            categorical_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("one_hot_encoder", OneHotEncoder()),
                    ("scaler", StandardScaler(with_mean=False))
                ]
            )
            
            logging.info("Numerical feature encoding done.")
            logging.info(f"Numerical features: {numerical_features}")
            logging.info("Categorical feature encoding done.")
            logging.info(f"Categorical features: {categorical_features}")
            
            preprocessor = ColumnTransformer(
                [
                ("numerical_pipeline", numerical_pipeline, numerical_features),
                ("categorical_pipeline", categorical_pipeline, categorical_features)
                ]
            )
            
            return preprocessor
            
        except Exception as e:
            raise CustomException(e, sys)
        
    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_data = pd.read_csv(train_path)
            test_data = pd.read_csv(test_path)
            
            logging.info("Training data and Test data read.")
            logging.info("Obtaining preprocessor object.")
            
            preprocessor_obj = self.get_data_transformer_obj()
            
            target_column = "math_score"
            numerical_features = ['reading_score', 'writing_score']
            
            input_feature_train_data = train_data.drop(columns=[target_column], axis=1)
            target_feature_train_data = train_data[target_column]
            input_feature_test_data = test_data.drop(columns=[target_column], axis=1)
            target_feature_test_data = test_data[target_column]
            
            logging.info(
                f"Performing preprocessing on train and test sets."
            )
            
            input_feature_train_arr = preprocessor_obj.fit_transform(input_feature_train_data)
            input_feature_test_arr = preprocessor_obj.transform(input_feature_test_data)
            
            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_data)]
            test_arr = np.c_[input_feature_test_arr, np.array (target_feature_test_data)]
            
            logging.info(f"Saved preprocessing object.")
            
            save_object(
                file_path = self.data_transformation_config.preprocessor_obj_file_path,
                obj = preprocessor_obj
            )
            
            return(
                train_arr, 
                test_arr, 
                self.data_transformation_config.preprocessor_obj_file_path
            )
            
        except Exception as e:
            raise CustomException(e, sys)