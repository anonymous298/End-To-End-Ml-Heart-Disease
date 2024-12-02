import os
import sys
import numpy as np             
import pandas as pd              

from src.exception import CustomException
from src.logger import get_logger
from src.utils import fetch_train_test_data, save_model

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

logger = get_logger('data preprocessing')

class DataPreprocessingConfig:
    preprocessor_path: str = os.path.join('model', 'preprocessor.pkl')

class DataPreprocessing:
    def __init__(self):
        self.data_preprocessing_config = DataPreprocessingConfig()

    def give_preprocessor_object(self, X):
        '''
        Create and return the preprocessor object.

        Parameters:
            X (dataframe): dataframe for extracting numerical and categorical columns.

        Returns:
            Preprocessor Object.
        '''
        try:
            logger.info('Initializing Preprocessor Pipeline')

            num_features = X.select_dtypes('number').columns
            cat_features = X.select_dtypes('object').columns

            logger.info('creating numerical features pipeline')
            num_preprocessor = Pipeline(
                steps=[
                    ('imputer', SimpleImputer(strategy='mean')),
                    ('scaler', StandardScaler())
                ]
            )

            logger.info('creating categorical features pipeline')
            cat_preprocessor = Pipeline(
                steps=[
                    ('imputer', SimpleImputer(strategy='most_frequent')),
                    ('ohe', OneHotEncoder())
                ]
            )

            logger.info('Creating Preprocessor Transformer')
            preprocessor = ColumnTransformer(
                [
                    ('num_preprocessor', num_preprocessor, num_features),
                    ('cat_preprocessor', cat_preprocessor, cat_features)
                ]
            )

            logger.info('Preprocessor Object Created Successfully')

            return preprocessor

        except Exception as e:
            logger.error(e)
            raise CustomException(e, sys)
        
    def initiate_preprocessing(self, train_path, test_path):
        '''
        fetches train and test data and returns clean X_train, X_test, y_train, y_test.

        Parameters:
            train_path (str): training path in which the training data is present.
            test_path (str): testing path in which the testing data is present.

        Returns:
            clean X_train, X_test, y_train, y_test.
        '''
        try:
            X_train, X_test, y_train, y_test = fetch_train_test_data(train_path, test_path)

            logger.info('Getting preprocessor object')
            preprocessor = self.give_preprocessor_object(X_train)

            logger.info('Applying transformation on X_train and X_test')
            X_train = preprocessor.fit_transform(X_train)
            X_test = preprocessor.transform(X_test)

            save_model(
                self.data_preprocessing_config.preprocessor_path,
                preprocessor
            )

            logger.info('Preprocessing Completed')

            return (
                X_train, 
                X_test,
                y_train,
                y_test
            )

        except Exception as e:
            logger.error(e)
            raise CustomException(e, sys)