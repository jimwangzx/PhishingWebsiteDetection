from datetime import datetime
from training_raw_data_validation.raw_validation import RawDataValidation
from training_data_insertion.db_operations import DBOperation
from training_data_transformation.data_transformation import DataTransform
from app_logging.logger import create_log


class TrainValidation:
    def __init__(self, path):
        self.raw_data = RawDataValidation(path)
        self.DataTransform = DataTransform()
        self.DBOperation = DBOperation()
        self.logger = create_log("Training_Logs/training_main", filemode='a+')

    def train_validation(self):
        try:
            self.logger.info('Start of validation on files for training')
            # extracting values from prediction schema
            LengthOfDateStampInFile, LengthOfTimeStampInFile, column_names, noofcolumns = self.raw_data.values_from_schema()
            # getting the regex defined to validate filename
            regex = self.raw_data.manual_regex_creation()
            # validating filename of prediction files
            self.raw_data.validation_file_name_raw(regex, LengthOfDateStampInFile, LengthOfTimeStampInFile)
            # validating column length in the file
            self.raw_data.validate_column_length(noofcolumns)
            # validating if any column has all values missing
            self.raw_data.validate_missing_values_in_whole_column()
            self.logger.info("Raw data validation is completed.")

            # self.logger.info("Starting data transformation")
            # # below function adds quotes to the '?' values in some columns.
            # self.DataTransform.add_quotes_to_string_values_in_column()
            # self.logger.info("Data transformation is completed.")

            self.logger.info("Creating training database and tables on the basis of given schema")
            # Create database with given name, if present open the connection
            # Create table with columns given in schema
            self.DBOperation.create_table_db('websites')
            self.logger.info("Table creation is completed.")
            self.logger.info("Insertion of data into the database is started.")
            # insert csv files in the table
            self.DBOperation.insert_into_table_good_data('websites')
            self.logger.info("Insertion into the database is completed.")
            self.logger.info("Deleting good data folder")
            # Delete the good data folder after loading files in table
            self.raw_data.delete_existing_good_data_training_folder()
            self.logger.info("Good_Data folder is deleted")
            self.logger.info("Moving bad files to archive and deleting Bad_Data folder")
            # Move the bad files to archive folder
            self.raw_data.move_bad_files_to_archive_bad()
            self.logger.info("Bad files are moved to archive. Bad folder is deleted.")
            self.logger.info("Validation operation is completed.")
            self.logger.info("Extracting csv file from the database")
            # export data in table to csvfile
            self.DBOperation.selecting_data_from_table_into_csv("websites")
        except Exception as e:
            raise e
