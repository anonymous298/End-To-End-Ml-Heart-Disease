import os
import sys
import numpy as np
import pandas as pd

from dataclasses import dataclass

from src.exception import CustomException
from src.logger import get_logger
from src.utils import load_model

logger = get_logger('prediction pipeline')

@dataclass 
class PredictionPipelineConfig:
    preprocessor_path: str = os.path.join('model', 'preprocessor.pkl')
    model_path: str = os.path.join('model', 'model.pkl')

class CustomData:
    def __init__(self, age, sex, chestpaintype, restingbp, cholesterol, fastingbs, restingecg, maxhr, exerciseengina, oldpeak, st_slope):
        self.age = age
        self.sex = sex
        self.chestpaintype = chestpaintype
        self.restingbp = restingbp
        self.cholesterol = cholesterol
        self.fastingbs = fastingbs
        self.restingecg = restingecg
        self.maxhr = maxhr
        self.exerciseengina = exerciseengina
        self.oldpeak = oldpeak
        self.st_slope = st_slope

    def conver_data_to_dataframe(self):
        '''
        Returns the dataframe from inputs the user gave
        '''
        try:
            logger.info('Converting inputs to dataframe')
            data = {
                "Age": [self.age],
                "Sex": [self.sex],
                "ChestPainType": [self.chestpaintype],
                "RestingBP": [self.restingbp],
                "Cholesterol": [self.cholesterol],
                "FastingBS": [self.fastingbs],
                "RestingECG": [self.restingecg],
                "MaxHR": [self.maxhr],
                "ExerciseAngina": [self.exerciseengina],
                "Oldpeak": [self.oldpeak],
                "ST_Slope": [self.st_slope]
            }

            return pd.DataFrame(data)
        
        except Exception as e:
            logger.error(e)
            raise CustomException(e, sys)

class PredictionPipeline:
    def __init__(self):
        self.prediction_pipeline_config = PredictionPipelineConfig()

    def predict(self, input_df):
        '''
        Predicts the output from the inputs the user give

        Parameters:
            input_df: input dataframe.

        Returns:
            model Prediction.
        '''
        logger.info('Started taking prediction')

        try:

            preprocessor = load_model(self.prediction_pipeline_config.preprocessor_path)
            model = load_model(self.prediction_pipeline_config.model_path)

            logger.info("Applying transformation and prediction")
            X = preprocessor.transform(input_df)
            prediction = model.predict(X)

            logger.info('Prediction Complete')

            return prediction

        except Exception as e:
            logger.error(e)
            raise CustomException(e, sys)
        
if __name__ == '__main__':
    customdata = CustomData(
            age=79,
            sex='M',
            chestpaintype='ASY',
            restingbp=130,
            cholesterol=283,
            fastingbs=0,
            restingecg='Normal',
            maxhr=170,
            exerciseengina='N',
            oldpeak=1.5,
            st_slope='Flat'
        )
    dataframe = customdata.conver_data_to_dataframe()
    # dataframe = dataframe.drop(columns=['Unnamed: 0'], errors='ignore')
    prediction_pipeline = PredictionPipeline()
    prediction = prediction_pipeline.predict(dataframe)
    print(prediction)