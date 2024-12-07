import os
import sys

from src.logger import get_logger

from src.components.data_ingestion import DataIngestion
from src.components.data_preprocessing import DataPreprocessing
from src.components.model_trainer import ModelTrainer
from src.components.model_evaluater import ModelEvaluation

logger = get_logger('training pipeline')

def training_pipeline():
    logger.info('Training Pipeline Initialized')
    data_ingest = DataIngestion()   # Creating DataIngestion Object
    train_path, test_path = data_ingest.load_data()  # Returns Train and Test path

    data_preprocessing = DataPreprocessing()  #Creating Data Preprocessing Object
    X_train, X_test, y_train, y_test = data_preprocessing.initiate_preprocessing(train_path, test_path)  #Getting X_train, X_test, y_train, y_test

    model_training = ModelTrainer()  # Creating ModelTrainer object
    trained_models = model_training.initiate_training(X_train, y_train)  # Training and getting our trained models

    model_evaluation = ModelEvaluation()  #Creating ModelEvalution object
    model_evaluation.evaluate(trained_models, X_train, X_test, y_train, y_test)   # Evaluating and saving our best model

    model_evaluation.hyperparameter_tuning(X_train, X_test, y_train, y_test)  # Hyperparameter tuning

    logger.info('Training Pipeline COmpleted')

if __name__ == '__main__':
    training_pipeline()

