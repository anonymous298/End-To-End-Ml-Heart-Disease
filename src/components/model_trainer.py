import os
import sys

from src.exception import CustomException
from src.logger import get_logger
from src.utils import get_models

logger = get_logger('model trainer')

class ModelTrainer:
    def __init__(self):
        self.models = get_models()

    def initiate_training(self, X_train, y_train):
        '''
        This will start training of the model and returns the trained model which will be evaluate.

        Parameters:
            X_train (DataFrame): X_train data for training models.
            y_train (DataFrame): y_train data for training models.

        Returns:
            Trained Models
            return_type: Dict.
        '''

        self.trained_models = {}

        try:
            logger.info('Starting Model training')

            for mod_name, model in self.models.items():
                logger.info(f"Training Model: {mod_name}")
                model = model
                model.fit(X_train, y_train)

                self.trained_models[mod_name] = model
                logger.info(f'{mod_name} Trained')

            logger.info('All Models training completed.')

            return self.trained_models

        except Exception as e:
            logger.error(e)
            raise CustomException(e, sys)