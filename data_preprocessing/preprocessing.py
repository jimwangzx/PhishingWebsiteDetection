import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler
# from sklearn.compose import ColumnTransformer
# from sklearn_pandas import CategoricalImputer


class Preprocessor:
    """Used to clean and transform the data before training
    """
    def __init__(self, file_object, logger_function):
        self.file_object = file_object
        self.logger = logger_function(file_object, filemode="a+")

    def remove_columns(self, data, columns):
        """Removes the given columns from a pandas dataframe

           Params:
           data: dataframe, data before column/columns removal
           columns: list, contains columns to remove

           Returns:
           A pandas DataFrame after removing the specified columns

           Raises:
           Exception in case of failure
        """
        self.logger.info('Entered the remove_columns method of the Preprocessor class')
        self.data = data
        self.columns = columns
        try:
            self.useful_data = self.data.drop(labels=self.columns, axis=1)
            self.logger.info('Column removal Successful.Exited the remove_columns method of \
            the Preprocessor class')
            return self.useful_data
        except Exception as e:
            self.logger.error(f'Exception occurred in remove_columns method of the Preprocessor class. \
            Exception message: {e}')
            self.logger.info('Column removal Unsuccessful. Exited the remove_columns method of \
            the Preprocessor class')
            raise Exception()

    def separate_label_feature(self, data, label_column_name):
        """Separates the features and a label column.
        
           Params:
           data: dataframe, data before separation
           label_column_name: str, name of target variable

           Returns:
           two separate Dataframes, one containing features and the other containing labels

           Raises:
           Exception in case of failure
        """
        self.logger.info('Entered the separate_label_feature method of the Preprocessor class')
        try:
            self.X = data.drop(labels=label_column_name, axis=1)
            self.y = data[label_column_name]
            self.logger.info('Label separation successful. Exited the separate_label_feature method of \
            the Preprocessor class')
            return self.X, self.y
        except Exception as e:
            self.logger.error(f'Exception occurred in separate_label_feature method of the Preprocessor class. \
            Exception message: {e}')
            self.logger.info(f'Label separation unsuccessful. Exited the separate_label_feature method of \
            the Preprocessor class')
            raise Exception()

    def drop_unnecessary_columns(self, data, column_name_list):
        """Drops the unwanted columns as discussed in EDA section.

           Params:
           data: dataframe, data before column(s) removal
           column_name_list: list, contains columns' names to drop

           Returns:
           data: dataframe, data after dropping the columns
        """
        data = data.drop(column_name_list, axis=1)
        return data

    def replace_invalid_values_with_null(self, data):
        """Replaces invalid values i.e. '?' with null

           Params:
           data: dataframe, data before invalid values' replacement

           Returns:
           data: dataframe, data after the replacement operation
        """
        for column in data.columns:
            count = data[column][data[column] == '?'].count()
            if count != 0:
                data[column] = data[column].replace('?', np.nan)
        return data

    def is_null_present(self, data):
        """Checks whether there are null values present in the pandas Dataframe or not

           Params:
           data: dataframe, data to check for null values

           Returns:
           True if null values are present in the DataFrame, False if they are not present and
           the list of columns for which null values are present.

           Raises:
           Exception in case of failure
        """
        self.logger.info('Entered the is_null_present method of the Preprocessor class')
        self.null_present = False
        self.cols_with_missing_values = []
        self.cols = data.columns
        try:
            self.null_counts = data.isna().sum()
            for i in range(len(self.null_counts)):
                if self.null_counts[i] > 0:
                    self.null_present = True
                    self.cols_with_missing_values.append(self.cols[i])
            if self.null_present:
                self.dataframe_with_null = pd.DataFrame()
                self.dataframe_with_null['columns'] = data.columns
                self.dataframe_with_null['missing values count'] = np.asarray(data.isna().sum())
                self.dataframe_with_null.to_csv('Preprocessing_Data/null_values.csv')
            self.logger.info('Finding missing values is a success.Data written to the null values file. \
            Exited the is_null_present method of the Preprocessor class')
            return self.null_present, self.cols_with_missing_values
        except Exception as e:
            self.logger.error(f'Exception occurred in is_null_present method of the Preprocessor class. \
            Exception message: {e}')
            self.logger.info('Finding missing values failed. Exited the is_null_present method \
            of the Preprocessor class')
            raise Exception()

    def encode_categorical_values(self, data, cat_col_list):
        """Encodes all the categorical values in the training set

           Params:
           data: dataframe, data before encoding
           cat_col_list: list, contains categorical columns to encode

           Returns:
           data: dataframe, data encoding the categorical values
        """
        for col in cat_col_list:
            data[col] = data[col].map({0: 'no', 1: 'yes', -1: 'missing'})

        data = pd.get_dummies(data, drop_first=True)
        return data

    def encode_categorical_values_prediction(self, data, cat_col_list):
        """Encodes all the categorical values in the prediction set

           Params:
           data: dataframe, data before encoding
           cat_col_list: list, contains categorical columns to encode

           Results:
           data: dataframe, data after encoding categorical values
        """
        for col in cat_col_list:
            data[col] = data[col].map({0: 'no', 1: 'yes', -1: 'missing'})
        data = pd.get_dummies(data)
        return data

    def impute_missing_values(self, data, cols_with_missing_values):
        """Replaces all the missing values in the Dataframe using KNN Imputer

           Params:
           data: dataframe, data before replacing missing values
           cols_with_missing_values: list, contains columns with missing values

           Results:
           data: dataframe, data after missing values' imputation

           Raises:
           Exception in case of failure
        """
        self.logger.info('Entered the impute_missing_values method of the Preprocessor class')
        self.data = data
        self.cols_with_missing_values = cols_with_missing_values
        try:
            self.imputer = KNNImputer()
            for col in self.cols_with_missing_values:
                self.data[col] = self.imputer.fit_transform(self.data[col])
            self.logger.info('Imputing missing values Successful. Exited the impute_missing_values \
            method of the Preprocessor class')
            return self.data
        except Exception as e:
            self.logger.error(f'Exception occurred in impute_missing_values method of the Preprocessor class. \
            Exception message: {e}')
            self.logger.info('Imputing missing values failed. Exited the impute_missing_values \
            method of the Preprocessor class')
            raise Exception()

    def get_columns_with_zero_std_deviation(self, data):
        """Finds out the columns which have a standard deviation of zero

           Params:
           data: dataframe, data to check for standard deviation

           Results:
           List of the columns with standard deviation of zero

           Raises:
           Exception in case of failure
        """
        self.logger.info('Entered the get_columns_with_zero_std_deviation method of the Preprocessor class')
        self.columns = data.columns
        self.data_n = data.describe()
        self.col_to_drop = []

        try:
            for x in self.columns:
                if self.data_n[x]['std'] == 0:
                    self.col_to_drop.append(x)
            self.logger.info('Column search for standard deviation of zero is successful. \
            Exited the get_columns_with_zero_std_deviation method of the Preprocessor class')
            return self.col_to_drop

        except Exception as e:
            self.logger.error(f'Exception occurred in get_columns_with_zero_std_deviation \
            method of the Preprocessor class. Exception message: {e}')
            self.logger.info('Column search for standard deviation of zero failed. \
            Exited the get_columns_with_zero_std_deviation method of the Preprocessor class')
            raise Exception()

    def standardize_data(self, data):
        """Standardizes numeric/continuous columns

           Params:
           data: dataframe, data before standardization

           Results:
           Standardized dataframe and standardization object

           Raises:
           Exception in case of failure
        """
        self.logger.info('Entered the standardize_data method of the Preprocessor class')
        try:
            cont_col_list = [col for col in data.columns if data[col].nunique() > 3]

            def log_transform(x):
                # add 1000 to x as it may contain values equal to -999
                return np.log(x + 1000)

            for col in cont_col_list:
                data[col] = data[col].apply(log_transform)

            scaler = StandardScaler()
            data[cont_col_list] = scaler.fit_transform(data[cont_col_list])
            # preproc = ColumnTransformer(
            #     transformers=[
            #         ('log_transform', FunctionTransformer(log_transform), cont_col_list),
            #         ('scale', StandardScaler(), cont_col_list),
            #     ],
            #     remainder="passthrough",
            # )
            # data = preproc.fit_transform(data)
            self.logger.info('The data has been standardized')
            return data, scaler
        except:
            self.logger.info("Error occurred in standardization process", exc_info=True)

    def replace_missing_values(self, data, cont_col_list):
        """Replaces -1 values with -999 values in continuous columns

           Params:
           data: dataframe, data before replacement of -1 values
           cont_col_list: list, contains continuous columns' names

           Results:
           data: dataframe, data with -1 values replaced with -999 in continuous columns

           Raises:
           Exception in case of failure
        """
        self.logger.info('Entered the replace_missing_values method of the Preprocessor class')
        try:
            for col in cont_col_list:
                data[col] = data[col].apply(lambda x: -999 if x == -1 else x)
            self.logger.info('The -1 values have been replaced with -999.')
            return data
        except:
            self.logger.info("Error occurred in replacement process", exc_info=True)


