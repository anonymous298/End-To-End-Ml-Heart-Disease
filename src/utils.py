import os
import sys
import numpy as np                    
import pandas as pd       
import dill         

from src.exception import CustomException
from src.logger import get_logger

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

logger = get_logger('utils')

def save_model(file_path, model):
    '''
    Save the desired model on desired path.

    Parameters:
        file_path (str): file path on which the model will be saved.
        model (object): model which will be saved on the path.

    Returns:
        None.
    '''
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        logger.info('saving model')
        with open(file_path, 'wb') as file:
            dill.dump(model, file)

        logger.info('saving completed')

    except Exception as e:
        logger.error(e)
        raise CustomException(e, sys)

def fetch_train_test_data(train_path, test_path):
    '''
    fetch train and test data from train and test path and returns the X_train, X_test, y_train, y_test.

    Parameters:
        train_path (str): train data path on which the train data is present
        test_path (str): test data path on which the test data is present

    Returns:
        X_train, X_test, y_train, y_test
    '''
    try:
        logger.info('fetching Train and Test data')
        train_data = pd.read_csv(train_path)
        test_data = pd.read_csv(test_path)

        DEPENDENT_FEATURE = 'HeartDisease'

        logger.info('Splitting data into training and testing split')
        X_train, X_test, y_train, y_test = (
            train_data.drop([DEPENDENT_FEATURE], axis=1),
            test_data.drop([DEPENDENT_FEATURE], axis=1),
            train_data[DEPENDENT_FEATURE],
            test_data[DEPENDENT_FEATURE]
        )

        logger.info('Fetching and splitting completed of train and test path')

        return (
            X_train,
            X_test, 
            y_train,
            y_test
        )

    except Exception as e:
        logger.error(e)
        raise CustomException(e, sys)
    
def get_models():
    '''
    has models used for classification task.

    Parameters:
        None

    Returns:
        bunch of Models.
    '''
    models = {
        'LogisticRegression' : LogisticRegression(),
        'SVC' : SVC(),
        'KNeighborsClassifier' : KNeighborsClassifier(),
        'DecisionTreeClassifier' : DecisionTreeClassifier(),
        'RandomForestClassifier' : RandomForestClassifier()
    }

    return models

def get_param_grid():
    '''
    Returns the param grid for each model.

    Returns:
        param grid.
    '''

    param_grid = {
    'LogisticRegression': {
        'penalty': ['l1', 'l2', 'elasticnet', 'none'],
        'C': [0.01, 0.1, 1, 10],
        'solver': ['lbfgs', 'saga'],  # Adjust based on penalties
        'max_iter': [100, 200, 500],
    },
    'SVC': {
        'C': [0.1, 1, 10, 100],
        'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
        'gamma': ['scale', 'auto'],
    },
    'KNeighborsClassifier': {
        'n_neighbors': [3, 5, 7, 9],
        'weights': ['uniform', 'distance'],
        'metric': ['euclidean', 'manhattan', 'minkowski'],
    },
    'DecisionTreeClassifier': {
        'criterion': ['gini', 'entropy'],
        'splitter': ['best', 'random'],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 5],
    },
    'RandomForestClassifier': {
        'n_estimators': [50, 100, 200],
        'criterion': ['gini', 'entropy'],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 5],
    }
}
    
    return param_grid