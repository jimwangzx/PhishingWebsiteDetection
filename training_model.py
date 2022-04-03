# This is the entry point for training machine learning models
# Doing the necessary imports
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from data_ingestion import data_loader
from data_preprocessing import preprocessing
from data_preprocessing import clustering
from best_model_finder import tuner
from file_operations import file_methods
from app_logging.logger import create_log
import pickle
import pandas as pd
import numpy as np


class TrainModel:

    def __init__(self):
        self.logger_function = create_log
        self.file_object = "Training_Logs/model_training"
        self.logger = create_log("Training_Logs/model_training", filemode="a+")

    def training_model(self):
        # Logging the start of training
        self.logger.info('Start of model training')
        try:
            # Getting the data from the source
            data_getter = data_loader.DataGetter(self.file_object, self.logger_function)
            data = data_getter.get_data()
            # Doing the data preprocessing
            preprocessor = preprocessing.Preprocessor(self.file_object, self.logger_function)

            # remove the unnamed column as it doesn't contribute to prediction.
            # data=preprocessor.remove_columns(data,['Wafer'])
            # getting columns with zero variance
            col_to_drop = preprocessor.get_columns_with_zero_std_deviation(data)
            # removing columns with zero variance
            data = preprocessor.drop_unnecessary_columns(data, col_to_drop)
            # replacing '?' values with np.nan as discussed in the EDA part
            # data = preprocessor.replace_invalid_values_with_null(data)

            # check if missing values are present in the dataset
            is_null_present, cols_with_missing_values = preprocessor.is_null_present(data)

            # if missing values are there, replace them appropriately.
            if is_null_present:
                # missing value imputation
                data = preprocessor.impute_missing_values(data, cols_with_missing_values)

            # create categorical columns' list
            cat_col_list = [col for col in data.columns if data[col].nunique() == 3 and col != 'phishing']
            # Saving categorical columns' list
            with open("EncoderPickle/cat_col_list.txt", 'wb') as file:
                pickle.dump(cat_col_list, file)

            # create continuous columns' list
            cont_col_list = [col for col in data.columns if col not in cat_col_list and col != 'phishing']

            # Saving continuous columns' list
            with open("EncoderPickle/cont_col_list.txt", 'wb') as file:
                pickle.dump(cont_col_list, file)

            # get encoded values for categorical data
            data = preprocessor.encode_categorical_values(data, cat_col_list)

            # Replacing -1 with -999 in continuous columns
            data = preprocessor.replace_missing_values(data, cont_col_list)

            # saving columns/features that will be used for training
            col_to_keep = list(data.columns)
            col_to_keep.remove('phishing')
            with open("EncoderPickle/col_to_keep.txt", 'wb') as file:
                pickle.dump(col_to_keep, file)

            # create separate features and labels
            X, y = preprocessor.separate_label_feature(data, label_column_name='phishing')

            # drop the columns obtained above
            # X = preprocessor.remove_columns(X, cols_to_drop)

            # Applying the clustering approach
            # Object initialization
            kmeans = clustering.KMeansClustering(self.file_object, self.logger_function)
            # Scaling continuous columns for kmeans algo
            scaler = StandardScaler()
            scaled_cont_X = pd.DataFrame(scaler.fit_transform(X[cont_col_list]))
            scaled_cont_X.columns = cont_col_list
            # Saving the kmeans' scaler
            with open("EncoderPickle/kmeans_scaler.txt", 'wb') as file:
                pickle.dump(scaler, file)
            # Using the elbow plot to find the number of optimum clusters
            number_of_clusters = kmeans.elbow_plot(scaled_cont_X)

            # Divide the data into clusters
            X['Cluster'] = kmeans.create_clusters(scaled_cont_X, number_of_clusters)
            # Create a new column in the dataset consisting of the corresponding cluster assignments.
            X['Labels'] = y

            # Getting the unique clusters from our dataset
            list_of_clusters = X['Cluster'].unique()

            # Initiating a dictionary to save dropped columns and standardization pipelines of each cluster
            drop_pipeline_dict = {}
            # Parsing all the clusters and looking for the best ML algorithm for each cluster
            for i in list_of_clusters:
                # Filter the data for one cluster
                cluster_data = X[X['Cluster'] == i]

                # Prepare the feature and Label columns
                cluster_features = cluster_data.drop(['Labels', 'Cluster'], axis=1)
                cluster_label = cluster_data['Labels']

                # Splitting the data into training and test set for each cluster one by one
                X_train, X_test, y_train, y_test = train_test_split(cluster_features,
                                                                    cluster_label,
                                                                    test_size=0.2,
                                                                    random_state=42,
                                                                    stratify=cluster_label)
                # Getting columns with zero variance inside a cluster
                cluster_col_to_drop = preprocessor.get_columns_with_zero_std_deviation(X_train)

                # Dropping columns with zero variance inside a cluster
                if len(cluster_col_to_drop) != 0:
                    X_train = X_train.drop(columns=cluster_col_to_drop)
                    X_test = X_test.drop(columns=cluster_col_to_drop)
                drop_pipeline_dict[i] = {}
                drop_pipeline_dict[i]['drop_cols'] = cluster_col_to_drop

                # Standardizing the data
                X_train, standardization_scaler = preprocessor.standardize_data(X_train)
                cont_col_list = [col for col in X_train.columns if X_train[col].nunique() > 3]
                drop_pipeline_dict[i]['cont_col'] = cont_col_list
                for col in cont_col_list:
                    X_test[col] = X_test[col].apply(lambda x: np.log(x + 1000))
                X_test[cont_col_list] = standardization_scaler.transform(X_test[cont_col_list])
                drop_pipeline_dict[i]['scaler'] = standardization_scaler

                # Object initialization
                model_finder = tuner.ModelFinder(self.file_object, self.logger_function)

                # Getting the best model for each of the clusters
                best_model_name, best_model = model_finder.get_best_model(X_train, y_train, X_test,  y_test)

                # Saving the best model to the directory.
                file_op = file_methods.FileOperation(self.file_object, self.logger_function)
                save_model = file_op.save_model(best_model, best_model_name+str(i))

            # Saving dropped columns and standardization pipelines of each cluster
            with open('EncoderPickle/drop_standardization_pipeline.txt', 'wb') as file:
                pickle.dump(drop_pipeline_dict, file)

            # Logging the successful Training
            self.logger.info('Successful end of training')

        except Exception:
            # Logging the unsuccessful Training
            self.logger.error('Unsuccessful end of training', exc_info=True)
            raise Exception
