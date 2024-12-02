import os
import sys
import numpy as np
import pandas as pd

from dataclasses import dataclass

from src.logger import get_logger
from src.exception import CustomException

from sklearn.model_selection import train_test_split

logger = get_logger('data_ingestion')

@dataclass
class DataIngestionConfig:
    raw_data_path: str = os.path.join('artifacts', 'data.csv')
    train_data_path: str = os.path.join('artifacts', 'train.csv')
    test_data_path: str = os.path.join('artifacts', 'test.csv')

class DataIngestion:
    def __init__(self):
        self.data_ingestion_config = DataIngestionConfig()

    def load_data(self):
        try:   
            logger.info('Reading data from source')
            df = pd.read_csv('Notebooks/data/heart.csv')

            logger.info(f'Saving raw data to path {self.data_ingestion_config.raw_data_path}')
            df.to_csv(self.data_ingestion_config.raw_data_path)

            logger.info('Splitting data into training and testing')
            train_data, test_data = train_test_split(df, test_size=0.2, random_state=42)

            logger.info('Saving train data to path', self.data_ingestion_config.train_data_path)
            train_data.to_csv(self.data_ingestion_config.train_data_path)

            logger.info('Saving test data to path', self.data_ingestion_config.test_data_path)
            test_data.to_csv(self.data_ingestion_config.test_data_path)

            logger.info('Data Fetching and Saving completed')

            return (
                self.data_ingestion_config.train_data_path,
                self.data_ingestion_config.test_data_path
            )

        except Exception as e:
            logger.error(e)
            raise CustomException(e, sys)