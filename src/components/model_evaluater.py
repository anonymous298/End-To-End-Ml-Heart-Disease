import os
import sys

from src.exception import CustomException
from src.logger import get_logger
from src.utils import save_model, get_models, get_param_grid

from dataclasses import dataclass

from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV

logger = get_logger('model evaluation')

@dataclass
class ModelEvaluationConfig:
    model_path: str = os.path.join('model', 'model.pkl')

class ModelEvaluation:
    def __init__(self):
        self.model_evaluation_config = ModelEvaluationConfig()

    def evaluate(self, models, X_train, X_test, y_train, y_test):
        '''
        Starts evaluating trained models on training and testing data.

        Parameters:
            models: trained models.
            X_train: X_train indpendent data.
            X_test: X_test indpendent data.
            y_train: y_train dependent data.
            y_test: y_test dependent data.

        Returns:
            None.
            saving model to path.
        '''
        
        model_train_score = {}
        model_test_score = {}

        try:
            logger.info('starting evaluating models')
            for model_name, model in models.items():
                model = model
                
                y_train_pred = model.predict(X_train)
                y_test_pred = model.predict(X_test)

                train_accuracy = accuracy_score(y_train, y_train_pred)
                logger.info(f'Evaluating training performance of {model_name} -> {train_accuracy}.')

                test_accuracy = accuracy_score(y_test, y_test_pred)
                logger.info(f'Evaluating testing performance of {model_name} -> {test_accuracy}.')

                model_train_score[model_name] = train_accuracy
                model_test_score[model_name] = test_accuracy

            print(f"Training Model Performance: {model_train_score}")
            print(f"Testing Model Performance: {model_test_score}")

            logger.info('Finding out the best model')
            best_model_score = max(sorted(list(model_test_score.values())))

            best_model_name = list(model_test_score.keys())[
                list(model_test_score.values()).index(best_model_score)
            ]
            logger.info(f'Best model found: {best_model_name}')

            best_model = models[best_model_name]

            if best_model_score < 0.6:
                raise CustomException('Best Model Not Found', sys)

            save_model(
                self.model_evaluation_config.model_path,
                best_model
            )

            logger.info('Evaluation Completed.')

        except Exception as e:
            logger.info(e)
            raise CustomException(e, sys)
        
    def hyperparameter_tuning(self, X_train, X_test, y_train, y_test):
        '''
        Starts HyperParameterTuning for all models with param grid and returns test model scores.

        Parameters:
            X_train: X_train indpendent data.
            X_test: X_test indpendent data.
            y_train: y_train dependent data.
            y_test: y_test dependent data.

        Returns:
            returns tuned models score
            Prints tuned models score.
        '''

        try:
            models = get_models()
            params = get_param_grid()
            tuned_model_score = {}

            logger.info('Start Tuning Each Model')
            for i in range(len(models)):
                model = list(models.values())[i]
                param = params[list(models.keys())[i]]

                logger.info(f'training our GridSearchCv on Model {model}')
                grid = GridSearchCV(
                    estimator=model,
                    param_grid=param,
                    verbose=3,
                    n_jobs=-1,
                    cv=3
                )

                grid.fit(X_train, y_train)

                logger.info('Training our model on best params')
                model.set_params(**grid.best_params_)
                model.fit(X_train, y_train)

                y_pred = model.predict(X_test)

                accuracy = accuracy_score(y_test, y_pred)

                tuned_model_score[list(models.keys())[i]] = accuracy
                logger.info(f'Training completed score is -> {accuracy}')

            print(f'Models Accuracy after tuning: {tuned_model_score}')

            logger.info('Finding out the best model')
            best_model_score = max(sorted(list(tuned_model_score.values())))

            best_model_name = list(tuned_model_score.keys())[
                list(tuned_model_score.values()).index(best_model_score)
            ]
            logger.info(f'Best model found: {best_model_name}')

            best_model = models[best_model_name]

            if best_model_score < 0.6:
                raise CustomException('Best Model Not Found', sys)

            save_model(
                self.model_evaluation_config.model_path,
                best_model
            )


            logger.info('Hyperparameter tuning Completed')

        except Exception as e:
            logger.error(e)
            raise CustomException(e, sys)
        