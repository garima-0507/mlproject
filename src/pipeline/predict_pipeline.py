import os
import sys
import pandas as pd
from joblib import load
from src.exception import CustomException
# from src.utils import load_object   # no longer needed for model loading

class PredictPipeline:
    def __init__(self):
        try:
            # load the full pipeline saved with joblib
            model_path = os.path.join("artifacts", "model.joblib")
            self.model = load(model_path)
        except Exception as e:
            # keep your CustomException usage
            raise CustomException(e, sys)

    def predict(self, features):
        try:
            # features must be a pandas DataFrame with the same column names used in training
            preds = self.model.predict(features)
            return preds

        except Exception as e:
            raise CustomException(e, sys)


class CustomData:
    def __init__(  self,
        gender: str,
        race_ethnicity: str,
        parental_level_of_education: str,
        lunch: str,
        test_preparation_course: str,
        reading_score: int,
        writing_score: int):

        self.gender = gender
        self.race_ethnicity = race_ethnicity
        self.parental_level_of_education = parental_level_of_education
        self.lunch = lunch
        self.test_preparation_course = test_preparation_course
        self.reading_score = reading_score
        self.writing_score = writing_score

    def get_data_as_data_frame(self):
        try:
            custom_data_input_dict = {
                "gender": [self.gender],
                "race_ethnicity": [self.race_ethnicity],
                "parental_level_of_education": [self.parental_level_of_education],
                "lunch": [self.lunch],
                "test_preparation_course": [self.test_preparation_course],
                "reading_score": [self.reading_score],
                "writing_score": [self.writing_score],
            }

            return pd.DataFrame(custom_data_input_dict)

        except Exception as e:
            raise CustomException(e, sys)
