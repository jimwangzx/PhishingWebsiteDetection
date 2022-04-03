import shutil
from cassandra.cluster import Cluster
from cassandra.auth import PlainTextAuthProvider
from os import listdir
import os
# import csv
from app_logging.logger import create_log
import pandas as pd
import re
from tqdm import tqdm


class DBOperation:
    """Handles all database operations
    """
    def __init__(self):
        # self.path = 'Prediction_Database/'
        self.bad_file_path = "Prediction_Raw_Files_Validated/Bad_Raw"
        self.good_file_path = "Prediction_Raw_Files_Validated/Good_Raw"
        self.data_base_connection_logger = create_log("Prediction_Logs/data_base_connection", filemode="a+")
        self.db_table_create_logger = create_log("Prediction_Logs/db_table_create", filemode="a+")
        self.db_insert_logger = create_log("Prediction_Logs/db_insert", filemode="a+")
        self.db_export_csv_logger = create_log("Prediction_Logs/export_to_csv", filemode="a+")

    def data_base_connection(self, keyspace_name):
        """Creates the database with the given name
           If database already exists then opens the connection to it.

           Params:
           keyspace_name: str, keyspace name in the cassandra database

           Returns:
           Connection to the database

           Raises:
           ConnectionError in case of failure
        """
        try:
            cloud_config = {
                'secure_connect_bundle': 'training_data_insertion/secure-connect-phishing.zip'
            }
            auth_provider = PlainTextAuthProvider('jfJwUfXjxIXkyOoCUJZZjtnh',
                                                  'UsKZTkEo8F7Wllcb_X9PYWGYt5ZRMj4dmcqxdI.va4TGzN-tcqPZZPhzs9xsZf2mFHr.8,ZNO1ynf5HLxF3zk.jWzcCNmjtJo3RAK6q9g_j4ZdDpPpFZlGuQ0G5YF_tC')
            cluster = Cluster(cloud=cloud_config, auth_provider=auth_provider)
            session = cluster.connect(keyspace_name)

            self.data_base_connection_logger.info(f"Opened {keyspace_name} keyspace successfully.")
        except ConnectionError:
            self.data_base_connection_logger.error(f"Error while connecting to database: {ConnectionError}")
            raise ConnectionError
        return session

    def create_table_db(self, keyspace_name):
        """Creates a table in the given database
           It will be used to insert the good data after the raw data validation.

           Params:
           keyspace_name: str, keyspace name in the cassandra database

           Returns:
           None

           Raises:
           Exception in case of failure
        """
        try:
            data = pd.read_csv('Prediction_Batch_Files/phishing_20032022_160100.csv')

            tuple_list = []
            for col in data.columns:
                tuple_list.append(' '.join([col, re.findall(r'\D+', str(data[col].dtype))[0]]))

            tuple_list1 = tuple_list[:56]
            tuple_list1.insert(0, 'website_id int')
            tuple_list1.append('primary key(website_id)')
            column_sequence1 = ', '.join(tuple_list1)

            tuple_list2 = tuple_list[56:]
            tuple_list2.insert(0, 'website_id int')
            tuple_list2.append('primary key(website_id)')
            column_sequence2 = ', '.join(tuple_list2)

            session = self.data_base_connection(keyspace_name)

            try:
                create_query = f"CREATE TABLE prediction_phishing1 ({column_sequence1});"
                session.execute(create_query)

                create_query = f"CREATE TABLE prediction_phishing2 ({column_sequence2});"
                session.execute(create_query)

                self.db_table_create_logger.info("Tables created successfully.")
            except:
                self.db_table_create_logger.info("Tables are in the database: no need to recreate them.", exc_info=True)

            session.shutdown()
            self.data_base_connection_logger.info(f"Closed {keyspace_name} keyspace successfully.")

        except Exception as e:
            self.db_table_create_logger.error(f"Error while creating tables: {e} ")
            raise e

    def insert_into_table_good_data(self, keyspace_name):
        """Inserts the Good data files from the Good_Raw folder into the above created tables

           Params:
           keyspace_name: str, keyspace name in the cassandra database

           Returns:
           None

           Raises:
           Exception in case of failure
        """
        session = self.data_base_connection(keyspace_name)
        good_file_path = self.good_file_path
        bad_file_path = self.bad_file_path
        only_files = [f for f in listdir(good_file_path)]

        try:
            data1 = pd.DataFrame(session.execute('SELECT * FROM "prediction_phishing1";').all())
            data2 = pd.DataFrame(session.execute('SELECT * FROM "prediction_phishing2";').all())
            data = pd.merge(data1, data2, on='website_id')
            data = data.drop(columns='website_id')
            n_rows = data.shape[0]
        except:
            n_rows = 0

        if n_rows == 0:
            for file in only_files:
                try:
                    f = open(good_file_path+'/'+file, 'r')
                    lines = f.readlines()

                    column_names1 = ','.join(lines[0].split(',')[:56])
                    column_names1 = 'website_id,' + column_names1

                    column_names2 = ','.join(lines[0].split(',')[56:])
                    column_names2 = 'website_id,' + column_names2

                    for i in tqdm(range(1, len(lines))):
                        column_values1 = ','.join(lines[i].split(',')[:56])
                        column_values1 = str(int(i - 1)) + ',' + column_values1

                        column_values2 = ','.join(lines[i].split(',')[56:])
                        column_values2 = str(int(i - 1)) + ',' + column_values2

                        insert_query1 = f"INSERT INTO prediction_phishing1 ({column_names1}) VALUES({column_values1});"
                        session.execute(insert_query1)

                        insert_query2 = f"INSERT INTO prediction_phishing2 ({column_names2}) VALUES({column_values2});"
                        session.execute(insert_query2)

                    f.close()
                    self.db_insert_logger.info(f" {file} file is loaded successfully.")
                except Exception as e:
                    self.db_insert_logger.error(f"Error while creating table: {e}")
                    shutil.move(good_file_path+'/' + file, bad_file_path)
                    self.db_insert_logger.info(f"{file} file is moved successfully.")
        else:
            self.db_insert_logger.info("The file has been inserted: no need to reinsert.")

        session.shutdown()

    def selecting_data_from_table_into_csv(self, keyspace_name):
        """Exports the data in GoodData table as a csv file

           Params:
           keyspace_name: str, keyspace name in the cassandra database

           Returns:
           None

           Raises:
           Exception in case of failure
        """

        self.file_from_db = 'Prediction_File_From_DB/'
        self.file_name = 'InputFile.csv'

        try:
            session = self.data_base_connection(keyspace_name)
            data1 = pd.DataFrame(session.execute('SELECT * FROM "prediction_phishing1";').all())
            data2 = pd.DataFrame(session.execute('SELECT * FROM "prediction_phishing2";').all())
            data = pd.merge(data1, data2, on='website_id')
            data = data.drop(columns='website_id')
            session.shutdown()

            if not os.path.isdir(self.file_from_db):
                os.makedirs(self.file_from_db)

            data.to_csv(self.file_from_db + self.file_name, index=False)

            self.db_export_csv_logger.info(f"The csv file is exported successfully.")

        except Exception as e:
            self.db_export_csv_logger.error(f"The csv file exporting failed. Error : {e}")