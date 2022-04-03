import pandas as pd
#from app_logging.logger import create_log


class DataGetter:
    """Used for obtaining the data from the source for training
    """
    def __init__(self, file_object, logger_function):
        self.training_file = 'Training_File_From_DB/InputFile.csv'
        self.file_object = file_object
        self.logger = logger_function(file_object, filemode="a+")

    def get_data(self):
        """Reads the data from source

           Returns:
           A pandas DataFrame.

           Raises:
           Exception in case of failure
        """
        self.logger.info('Entered the get_data method of the DataGetter class')
        try:
            self.data= pd.read_csv(self.training_file)
            self.logger.info('Data Load Successful. Exited the get_data method of the DataGetter class')
            return self.data
        except Exception as e:
            self.logger.error(f'Exception occurred in get_data method of the DataGetter class. Exception message: {e}')
            self.logger.info('Data Load Unsuccessful.Exited the get_data method of the DataGetter class')
            raise Exception()


