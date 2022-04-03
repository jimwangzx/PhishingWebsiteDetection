# import sqlite3
from datetime import datetime
from os import listdir
import os
import re
import json
import shutil
import pandas as pd
from app_logging.logger import create_log


class PredictionDataValidation:
    """Handles all the validation done on the raw prediction data
    """
    def __init__(self, path):
        self.batch_directory = path
        self.schema_path = 'schema_prediction.json'
        self.logger = create_log('Prediction_Logs/values_from_schema_validation', filemode='a+')
        self.general_logger = create_log("Prediction_Logs/general", filemode="a+")
        self.name_validation_logger = create_log("Prediction_Logs/name_validation", filemode="a+")
        self.column_validation_logger = create_log("Prediction_Logs/column_validation", filemode="a+")
        self.missing_value_logger = create_log("Prediction_Logs/missing_values_in_column", filemode="a+")

    def values_from_schema(self):
        """Extracts all the relevant information from the pre-defined "schema" file.

           Returns:
           LengthOfDateStampInFile: numeric, length of date stamp in a file name,
           LengthOfTimeStampInFile: numeric, length of time stamp in a file name,
           column_names: str, columns' names in a file,
           Number of Columns: numeric, number of columns in a file

           Raises:
           ValueError, KeyError, Exception in case of failure
        """

        try:
            with open(self.schema_path, 'r') as f:
                dic = json.load(f)
                f.close()
            pattern = dic['SampleFileName']
            LengthOfDateStampInFile = dic['LengthOfDateStampInFile']
            LengthOfTimeStampInFile = dic['LengthOfTimeStampInFile']
            column_names = dic['ColName']
            NumberofColumns = dic['NumberofColumns']

            message = f'''LengthOfDateStampInFile:: {LengthOfDateStampInFile} 
                          LengthOfTimeStampInFile:: {LengthOfTimeStampInFile} 
                          NumberofColumns:: {NumberofColumns}'''
            self.logger.info(message)
        except ValueError:
            self.logger.error("ValueError: Value not found inside schema_training.json")
            raise ValueError
        except KeyError:
            self.logger.error("KeyError: Key value error incorrect key passed")
            raise KeyError
        except Exception as e:
            self.logger.error(str(e))
            raise e

        return LengthOfDateStampInFile, LengthOfTimeStampInFile, column_names, NumberofColumns

    def manual_regex_creation(self):
        """Contains a manually defined regex based on the file name given in "schema" file.
           This regex is used to validate the filename of the prediction data.

           Returns:
           regex: regex pattern
        """
        regex = "['phishing']+['\_'']+[\d_]+[\d]+\.csv"
        return regex

    def create_directory_for_good_bad_raw_data(self):
        """Creates directories to store the good data and bad data after validating the prediction data

           Returns:
           None

           Raises:
           OSError in case of failure
        """
        try:
            path = os.path.join("Prediction_Raw_Files_Validated/", "Good_Raw/")
            if not os.path.isdir(path):
                os.makedirs(path)
            path = os.path.join("Prediction_Raw_Files_Validated/", "Bad_Raw/")
            if not os.path.isdir(path):
                os.makedirs(path)
        except OSError as ex:
            self.general_logger.error(f"Error while creating Directory: {ex}")
            raise OSError

    def delete_existing_good_data_training_folder(self):
        """Deletes the directory made to store the Good Data after loading the data in the table
           Once the good files are loaded in the DB, deleting the directory ensures space optimization.

           Returns:
           None

           Raises:
           OSError in case of failure
        """

        try:
            path = 'Prediction_Raw_Files_Validated/'
            if os.path.isdir(path + 'Good_Raw/'):
                shutil.rmtree(path + 'Good_Raw/')
                self.general_logger.info("GoodRaw directory deleted successfully.")
        except OSError as s:
            self.general_logger.error(f"Error while Deleting Directory: {s}")
            raise OSError

    def delete_existing_bad_data_training_folder(self):
        """Deletes the directory made to store the bad data

           Returns:
           None

           Raises:
           OSError in case of failure
        """
        try:
            path = 'Prediction_Raw_Files_Validated/'
            if os.path.isdir(path + 'Bad_Raw/'):
                shutil.rmtree(path + 'Bad_Raw/')
                self.general_logger.info("Bad_Raw directory deleted before starting validation.")
        except OSError as s:
            self.general_logger.error(f"Error while Deleting Directory : {s}")
            raise OSError

    def move_bad_files_to_archive_bad(self):
        """Deletes the directory made  to store the Bad Data after moving the data in an archive folder
           We archive the bad files to send them back to the user for invalid data issue.

           Returns:
           None

           Raises:
           OSError in case of failure
        """
        now = datetime.now()
        date = now.date()
        time = now.strftime("%H%M%S")
        try:
            source = 'Prediction_Raw_Files_Validated/Bad_Raw/'
            if os.path.isdir(source):
                path = "Prediction_Archive_Bad_Data"
                if not os.path.isdir(path):
                    os.makedirs(path)
                destination = 'Prediction_Archive_Bad_Data/BadData_' + str(date)+"_"+str(time)
                if not os.path.isdir(destination):
                    os.makedirs(destination)
                files = os.listdir(source)
                for f in files:
                    if f not in os.listdir(destination):
                        shutil.move(source + f, destination)
                self.general_logger.info("Bad files moved to archive")
                path = 'Prediction_Raw_Files_Validated/'
                if os.path.isdir(path + 'Bad_Raw/'):
                    shutil.rmtree(path + 'Bad_Raw/')
                self.general_logger.info("Bad Raw Data Folder Deleted successfully.")
        except Exception as e:
            self.general_logger.error(f"Error while moving bad files to archive: {e}")
            raise e

    def validation_file_name_raw(self, regex, LengthOfDateStampInFile, LengthOfTimeStampInFile):
        """Validates the name of the prediction csv files as per given name in the schema
           Regex pattern is used to do the validation. If name format do not match the file is moved
           to bad raw data folder else in good raw data.

           Params:
           regex: regex pattern
           LengthOfDateStampInFile: int, length of date stamp in filename
           LengthOfTimeStampInFile: int, length of time stamp in filename

           Returns:
           None

           Raises:
           Exception in case of failure
        """
        self.delete_existing_bad_data_training_folder()
        self.delete_existing_good_data_training_folder()

        only_files = [f for f in listdir(self.batch_directory)]

        try:
            self.create_directory_for_good_bad_raw_data()
            for filename in only_files:
                if re.match(regex, filename):
                    split_at_dot = re.split('.csv', filename)
                    split_at_dot = (re.split('_', split_at_dot[0]))
                    if len(split_at_dot[1]) == LengthOfDateStampInFile:
                        if len(split_at_dot[2]) == LengthOfTimeStampInFile:
                            shutil.copy("Prediction_Batch_Files/" + filename, "Prediction_Raw_Files_Validated/Good_Raw")
                            self.name_validation_logger.info(f"Valid File name. {filename} moved to Good_Raw folder.")
                        else:
                            shutil.copy("Prediction_Batch_Files/" + filename, "Prediction_Raw_Files_Validated/Bad_Raw")
                            self.name_validation_logger.info(f"Invalid File Name. {filename} moved to Bad_Raw folder.")
                    else:
                        shutil.copy("Prediction_Batch_Files/" + filename, "Prediction_Raw_Files_Validated/Bad_Raw")
                        self.name_validation_logger.info(f"Invalid File Name. {filename} moved to Bad_Raw folder.")
                else:
                    shutil.copy("Prediction_Batch_Files/" + filename, "Prediction_Raw_Files_Validated/Bad_Raw")
                    self.name_validation_logger.log(f"Invalid File Name. {filename} moved to Bad_Raw folder.")
        except Exception as e:
            self.name_validation_logger.error(f"Error occurred while validating FileName: {e}")
            raise e

    def validate_column_length(self, NumberofColumns):
        """Validates the number of columns in the csv file/files.
           It should be the same as given in the schema file. If not
           it is not suitable for processing and thus is moved to bad raw data folder.
           If the column number matches, file is kept in good raw data for processing.

           Params:
           NumberofColumns: int, number of columns in a csv file

           Returns:
           None

           Raises:
           Exception in case of failure
        """
        try:
            self.column_validation_logger.info("Column length validation has been started.")
            for file in listdir('Prediction_Raw_Files_Validated/Good_Raw/'):
                csv = pd.read_csv("Prediction_Raw_Files_Validated/Good_Raw/" + file)
                if csv.shape[1] == NumberofColumns:
                    pass
                else:
                    shutil.move("Prediction_Raw_Files_Validated/Good_Raw/" + file,
                                "Prediction_Raw_Files_Validated/Bad_Raw")
                    self.column_validation_logger.info(f"Invalid column length. {file} moved to Bad_Raw folder")
            self.column_validation_logger.info("Column length validation has been completed.")
        except OSError:
            self.column_validation_logger.error(f"Error occurred while moving the file: {OSError}")
            raise OSError
        except Exception as e:
            self.column_validation_logger.error(f"Error occurred: {e}")
            raise e

    def delete_prediction_file(self):
        if os.path.exists('Prediction_Output_File/Predictions.csv'):
            os.remove('Prediction_Output_File/Predictions.csv')

    def validate_missing_values_in_whole_column(self):
        """Checks for values missing in the columns.
           If all the values are missing, the file is not suitable for processing.
           Such files are moved to bad raw data.

           Returns:
           None

           Raises:
           Exception in case of failure
        """
        try:
            self.missing_value_logger.info("Missing values validation has been started.")

            for file in listdir('Prediction_Raw_Files_Validated/Good_Raw/'):
                csv = pd.read_csv("Prediction_Raw_Files_Validated/Good_Raw/" + file)
                count = 0
                for columns in csv:
                    if (len(csv[columns]) - csv[columns].count()) == len(csv[columns]):
                        count += 1
                        shutil.move("Prediction_Raw_Files_Validated/Good_Raw/" + file,
                                    "Prediction_Raw_Files_Validated/Bad_Raw")
                        self.missing_value_logger.info(f"{columns} is completely missing. {file} moved to Bad_Raw folder")
                        break
                if count == 0:
                    csv.to_csv("Prediction_Raw_Files_Validated/Good_Raw/" + file, index=None, header=True)
        except OSError:
            self.missing_value_logger.error(f"Error occurred while moving the file: {OSError}")
            raise OSError
        except Exception as e:
            self.logger.log(f"Error occurred: {e}")
            raise e
