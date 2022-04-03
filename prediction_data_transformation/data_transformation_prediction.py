# from datetime import datetime
from os import listdir
import pandas as pd
from app_logging.logger import create_log


class DataTransformPredict:
    """Used for transforming the good raw training data before loading it in database
    """
    def __init__(self):
        self.good_data_path = "Prediction_Raw_Files_Validated/Good_Raw"
        self.data_transform_logger = create_log("Prediction_Logs/data_transform", filemode="a+")

    def add_quotes_to_string_values_in_column(self):
        """Adds quotes to ? values in columns
        """
        try:
            only_files = [f for f in listdir(self.good_data_path)]
            for file in only_files:
                data = pd.read_csv(self.good_data_path + "/" + file)
                for column in data.columns:
                    count = data[column][data[column] == '?'].count()
                    if count != 0:
                         data[column] = data[column].replace('?', "'?'")
                data.to_csv(self.good_data_path + "/" + file, index=None, header=True)
                self.data_transform_logger.info(f"{file}: quotes added successfully")
        except Exception as e:
            self.data_transform_logger.error(f"Data Transformation failed because: {e}")
            raise e
