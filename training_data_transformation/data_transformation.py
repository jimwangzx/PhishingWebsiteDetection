# from datetime import datetime
from os import listdir
from app_logging.logger import create_log
import pandas as pd


class DataTransform:
    """ Transforms the good raw training data
         before loading it into a database
    """
    def __init__(self):
        self.good_data_path = "Training_Raw_Files_Validated/Good_Raw"
        self.logger = create_log("Training_Logs/add_quotes_to_string_values_in_column", filemode="a+")

    def add_quotes_to_string_values_in_column(self):
        """Converts all the columns with string datatype such that
           each value for that column is enclosed in quotes. This is done
           to avoid the error while inserting string values in table as varchar
        """
        try:
            only_files = [f for f in listdir(self.good_data_path)]
            for file in only_files:
                data = pd.read_csv(self.good_data_path+"/" + file)
                for column in data.columns:
                    count = data[column][data[column] == '?'].count()
                    if count != 0:
                        data[column] = data[column].replace('?', "'?'")
                data.to_csv(self.good_data_path + "/" + file, index=None, header=True)
                self.logger.info(f"Quotes added successfully to {file}.")
        except Exception as e:
            self.logger.error(f"Data transformation failed because: {e}")
