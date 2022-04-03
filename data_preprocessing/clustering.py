import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from kneed import KneeLocator
from file_operations import file_methods


class KMeansClustering:
    """Used to divide the data into clusters before training
    """
    def __init__(self, file_object, logger_function):
        self.file_object = file_object
        self.logger = logger_function(file_object, filemode="a+")
        self.logger_function = logger_function

    def elbow_plot(self, data):
        """Saves the plot (elbow plot) to decide the optimal number of clusters to the file

           Params:
           data: dataframe, includes features for cluster construction

           Results:
           A picture saved to the directory

           Raises:
           Exception in case of failure
        """
        self.logger.info('Entered the elbow_plot method of the KMeansClustering class')

        wcss = []
        try:
            for i in range(1, 11):
                kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
                kmeans.fit(data)
                wcss.append(kmeans.inertia_)
            plt.plot(range(1, 11), wcss)
            plt.title('The Elbow Method')
            plt.xlabel('Number of clusters')
            plt.ylabel('WCSS')
            # plt.show()
            plt.savefig('Preprocessing_Data/K-Means_Elbow.PNG')

            # Finding the value of the optimum cluster programmatically
            self.kn = KneeLocator(range(1, 11), wcss, curve='convex', direction='decreasing')
            self.logger.info(f'The optimum number of clusters is {self.kn.knee}. \
            Exited the elbow_plot method of the KMeansClustering class')
            return self.kn.knee
        except Exception as e:
            self.logger.error(f'Exception occurred in elbow_plot method of the KMeansClustering class. \
            Exception message: {e}')
            self.logger.info('Finding the number of clusters failed. \
            Exited the elbow_plot method of the KMeansClustering class')
            raise Exception()

    def create_clusters(self, data, number_of_clusters):
        """Augments the data with the cluster information

           Params:
           data: dataframe, includes features for cluster construction
           number_of_clusters: int, number of clusters to create

           Results:
           A dataframe with cluster column

           Raises:
           Exception in case of failure
        """
        self.logger.info('Entered the create_clusters method of the KMeansClustering class')

        self.data = data

        try:
            self.kmeans = KMeans(n_clusters=number_of_clusters, init='k-means++', random_state=42)
            # self.data = self.data[~self.data.isin([np.nan, np.inf, -np.inf]).any(1)]
            self.y_kmeans = self.kmeans.fit_predict(data)
            self.file_op = file_methods.FileOperation(self.file_object, self.logger_function)
            self.save_model = self.file_op.save_model(self.kmeans, 'KMeans')
            self.data['Cluster'] = self.y_kmeans
            self.logger.info(f'Successfully created {self.kn.knee} clusters. \
            Exited the create_clusters method of the KMeansClustering class')
            return self.data['Cluster']
        except Exception as e:
            self.logger.error(f'Exception occurred in create_clusters method of the KMeansClustering class. \
            Exception message: {e}')
            self.logger.info('Fitting the data to clusters failed. Exited the create_clusters method of \
            the KMeansClustering class')
            raise Exception()

