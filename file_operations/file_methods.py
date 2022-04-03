import pickle
import os
import shutil


class FileOperation:
    """Used to save the model after training and load the saved model for prediction
    """
    def __init__(self, file_object, logger_function):
        self.file_object = file_object
        self.logger = logger_function(file_object, filemode="a+")
        self.model_directory = 'Models/'

    def save_model(self, model, filename):
        """Saves the model file to directory

           Params:
           model: object, trained model
           filename: str, name for the trained model to save

           Returns:
           File gets saved

           Raises:
           Exception in case of failure
        """
        self.logger.info('Entered the save_model method of the FileOperation class')
        try:
            path = os.path.join(self.model_directory, filename)
            if os.path.isdir(path):
                shutil.rmtree(self.model_directory)
                os.makedirs(path)
            else:
                os.makedirs(path)
            with open(path +'/' + filename+'.sav', 'wb') as file:
                pickle.dump(model, file)
            self.logger.info(f'Model File {filename} saved. Exited the save_model method of the FileOperation class')

            return 'success'
        except Exception as e:
            self.logger.error(f'Exception occurred in save_model method of the FileOperation class. Exception message: {e}')
            self.logger.info('Model file {filename} could not be saved. Exited the save_model method of the FileOperation class')
            raise Exception()

    def load_model(self, filename):
        """Loads the model file to memory

           Params:
           filename: str, name of a trained model

           Returns:
           The Model file loaded in memory

           Raises:
           Exception in case of failure
        """
        self.logger.info('Entered the load_model method of the FileOperation class')
        try:
            with open(self.model_directory + filename + '/' + filename + '.sav', 'rb') as file:
                self.logger.info(f'Model file {filename} loaded. Exited the load_model method of the FileOperation class')
                return pickle.load(file)
        except Exception as e:
            self.logger.error(f'Exception occurred in load_model method of the FileOperation class. Exception message: {e}')
            self.logger.info(f'Model file {filename} could not be saved. Exited the load_model method of the FileOperation class')
            raise Exception()

    def find_correct_model_file(self, cluster_number):
        """Selects the correct model based on cluster number

           Params:
           cluster_number: int, cluster number

           Returns:
           The Model file

           Raises:
           Exception in case of failure
        """
        self.logger.info('Entered the find_correct_model_file method of the FileOperation class')
        try:
            self.cluster_number = cluster_number
            self.folder_name = self.model_directory
            self.list_of_model_files = []
            self.list_of_files = os.listdir(self.folder_name)
            for self.file in self.list_of_files:
                try:
                    if self.file.index(str(self.cluster_number)) != -1:
                        self.model_name = self.file
                except:
                    continue
            self.model_name = self.model_name.split('.')[0]
            self.logger.info('Exited the find_correct_model_file method of the FileOperation class.')
            return self.model_name
        except Exception as e:
            self.logger.error(f'Exception occurred in find_correct_model_file method of the FileOperation class. Exception message: {e}')
            self.logger.info('Exited the find_correct_model_file method of the FileOperation class with Failure')
            raise Exception()