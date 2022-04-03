import pandas as pd
from file_operations import file_methods
from data_preprocessing import preprocessing
from data_ingestion import data_loader_prediction
from app_logging.logger import create_log
from prediction_raw_data_validation.prediction_raw_validation import PredictionDataValidation
import pickle
import numpy as np


class Prediction:

    def __init__(self, path):
        self.file_object = "Prediction_Logs/prediction"
        self.log_function = create_log
        self.logger = create_log("Prediction_Logs/prediction", filemode="a+")
        self.prediction_data_val = PredictionDataValidation(path)

    def prediction_from_model(self):

        try:
            # deletes the existing prediction file from last run
            self.prediction_data_val.delete_prediction_file()
            self.logger.info('Start of prediction')
            data_getter = data_loader_prediction.DataGetterPred(self.file_object, self.log_function)
            data = data_getter.get_data()

            preprocessor = preprocessing.Preprocessor(self.file_object, self.log_function)

            # data = preprocessor.drop_unnecessary_columns(data, ['veiltype'])
            # replacing '?' values with np.nan as discussed in the EDA part
            # data = preprocessor.replace_invalid_values_with_null(data)

            is_null_present, cols_with_missing_values = preprocessor.is_null_present(data)
            if is_null_present:
                data = preprocessor.impute_missing_values(data, cols_with_missing_values)

            # Loading categorical columns' list
            with open("EncoderPickle/cat_col_list.txt", 'rb') as file:
                cat_col_list = pickle.load(file)

            # Loading continuous columns' list
            #cont_col_list = [col for col in data.columns if col not in cat_col_list and col != 'phishing']
            with open("EncoderPickle/cont_col_list.txt", 'rb') as file:
                cont_col_list = pickle.load(file)

            # get encoded values for categorical data
            data = preprocessor.encode_categorical_values_prediction(data, cat_col_list)

            # Replacing -1 with -999 in continuous columns
            data = preprocessor.replace_missing_values(data, cont_col_list)

            # data=data.to_numpy()
            file_loader = file_methods.FileOperation(self.file_object, self.log_function)
            kmeans = file_loader.load_model('KMeans')

            # create separate features and labels
            X, y = preprocessor.separate_label_feature(data, label_column_name='phishing')
            # Loading the kmeans' scaler
            with open("EncoderPickle/kmeans_scaler.txt", 'rb') as file:
                scaler = pickle.load(file)
            # Scaling continuous columns for kmeans algo
            scaled_cont_X = scaler.transform(X[cont_col_list])
            # Creating clusters
            clusters = kmeans.predict(scaled_cont_X)
            data['Clusters'] = clusters
            clusters = data['Clusters'].unique()

            # initialize blank list for storing predictions
            result = []
            # Let's load the encoder pickle file to decode the values
            # with open('EncoderPickle/enc.pickle', 'rb') as file:
            #     encoder = pickle.load(file)

            # loading columns/features that has been used for training
            with open("EncoderPickle/col_to_keep.txt", 'rb') as file:
                col_to_keep = pickle.load(file)

            # loading a dictionary containing dropped columns and
            # standardization pipelines of each cluster derived during training
            with open('EncoderPickle/drop_standardization_pipeline.txt', 'rb') as file:
                drop_pipeline_dict = pickle.load(file)

            # adding missing columns from col_to_keep as some categorical features
            # may not contain all the possible values available in training set
            add_col_list = [col for col in col_to_keep if col not in data.columns]
            if len(add_col_list) != 0:
                for col in add_col_list:
                    data[col] = 0

            for i in clusters:
                cluster_data = data[data['Clusters'] == i]
                # cluster_data = cluster_data.drop(['Clusters'], axis=1)
                cluster_data = cluster_data[col_to_keep]
                if len(drop_pipeline_dict[i]['drop_cols']) != 0:
                    cluster_data = cluster_data.drop(columns=drop_pipeline_dict[i]['drop_cols'])
                cont_col_list = drop_pipeline_dict[i]['cont_col']
                for col in cont_col_list:
                    cluster_data[col] = cluster_data[col].apply(lambda x: np.log(x + 1000))
                cluster_data[cont_col_list] = drop_pipeline_dict[i]['scaler'].transform(cluster_data[cont_col_list])
                model_name = file_loader.find_correct_model_file(i)
                model = file_loader.load_model(model_name)
                for val in model.predict(cluster_data):
                    result.append(val)
            result = pd.DataFrame(result, columns=['Predictions'])
            path = "Prediction_Output_File/Predictions.csv"
            # Append the result to the prediction file
            result.to_csv("Prediction_Output_File/Predictions.csv", header=True)
            self.logger.info('End of prediction')
        except Exception as ex:
            self.logger.error(f'Error occurred while running the prediction!! Error: {ex}')
            raise ex
        return path


