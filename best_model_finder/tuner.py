from mixed_naive_bayes import MixedNB
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
# from sklearn.naive_bayes import GaussianNB, BernoulliNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
# from sklearn.model_selection import cross_val_score
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
# from catboost import CatBoostClassifier
# from bayes_opt import BayesianOptimization
from sklearn.metrics import roc_auc_score, accuracy_score
from skopt import BayesSearchCV
from skopt.space import Real, Categorical, Integer
# import pandas as pd


class ModelFinder:
    """Used to find a model with the best accuracy and ROC-AUC score.
    """
    def __init__(self, file_object, logger_function):
        self.file_object = file_object
        self.logger = logger_function(file_object, filemode='a+')
        self.sv_classifier = SVC()
        self.xgb_classifier = XGBClassifier(objective='binary:logistic')
        self.logreg = LogisticRegression()
        self.rf = RandomForestClassifier(random_state=42)
        self.lgbm = LGBMClassifier(
                        boosting_type='dart',
                        objective='binary',
                        metric='binary_logloss',
                        max_depth=-1,
                        subsample_freq=0,
                        min_split_gain=0,
                        random_state=42
                    )
        # self.cb = CatBoostClassifier()

    def get_best_params_for_svm(self, train_x, train_y):
        """Finds the parameters for the SVM algorithm which give the best accuracy

           Params:
           train_x: dataframe, training features
           train_y: series, training target (1 if phishing is true and 0 otherwise)

           Returns:
           The model with the best parameters

           Raises:
           Exception in case of failure
        """
        self.logger.info('Entered the get_best_params_for_svm method of the ModelFinder class')

        try:
            self.param_grid = {
                "kernel": ['rbf', 'sigmoid'],
                "C": [0.1, 0.5, 1.0],
                "random_state": [0, 100, 200, 300]
            }
            self.grid = GridSearchCV(estimator=self.sv_classifier, param_grid=self.param_grid, 
                                     cv=5,  verbose=3, scoring='roc_auc')
            self.grid.fit(train_x, train_y)
            self.kernel = self.grid.best_params_['kernel']
            self.C = self.grid.best_params_['C']
            self.random_state = self.grid.best_params_['random_state']

            self.sv_classifier = SVC(kernel=self.kernel, C=self.C, random_state=self.random_state)
            self.sv_classifier.fit(train_x, train_y)
            self.logger.info(f'SVM best params: {self.grid.best_params_} \
            Exited the get_best_params_for_svm method of the ModelFinder class')
            return self.sv_classifier
        except Exception as e:
            self.logger.error(f'Exception occured in get_best_params_for_svm method of the ModelFinder class. \
            Exception message: {e}')
            self.logger.info('SVM training failed. Exited the get_best_params_for_svm method of \
            the ModelFinder class')
            raise Exception()

    def get_best_params_for_xgboost(self, train_x, train_y):
        """Finds parameters for XGBoost algorithm which give the best accuracy

           Params:
           train_x: dataframe, training features
           train_y: series, training target (1 if phishing is true and 0 otherwise)

           Returns:
           The model with the best parameters

           Raises:
           Exception in case of failure
        """
        self.logger.info('Entered the get_best_params_for_xgboost method of the ModelFinder class')

        try:
            self.param_grid_xgboost = {
                "n_estimators": [100, 130],
                "criterion": ['gini', 'entropy'],
                "max_depth": range(8, 10, 1)
            }
            self.grid= GridSearchCV(self.xgb_classifier, self.param_grid_xgboost, 
                                    verbose=3, cv=5, scoring='roc_auc')
            self.grid.fit(train_x, train_y)
            self.criterion = self.grid.best_params_['criterion']
            self.max_depth = self.grid.best_params_['max_depth']
            self.n_estimators = self.grid.best_params_['n_estimators']

            self.xgb_classifier = XGBClassifier(objective='binary:logistic', criterion=self.criterion, 
                                                max_depth=self.max_depth, n_estimators=self.n_estimators, 
                                                n_jobs=-1)
            self.xgb_classifier.fit(train_x, train_y)
            self.logger.info(f'XGBoost best params: {self.grid.best_params_} \
            Exited the get_best_params_for_xgboost method of the ModelFinder class')
            return self.xgb_classifier
        except Exception as e:
            self.logger.error(f'Exception occured in get_best_params_for_xgboost method of the ModelFinder class. \
            Exception message: {e}')
            self.logger.info('XGBoost parameter tuning failed. \
            Exited the get_best_params_for_xgboost method of the Model_Finder class')
            raise Exception()

    def get_best_params_for_logreg(self, train_x, train_y):
        """Finds the parameters for the LogisticRegression algorithm which give the best accuracy

           Params:
           train_x: dataframe, training features
           train_y: series, training target (1 if phishing is true and 0 otherwise)

           Returns:
           The model with the best parameters

           Raises:
           Exception in case of failure
        """
        self.logger.info('Entered the get_best_params_for_logreg method of the ModelFinder class')

        try:
            self.param_grid = {
                "C": [0.01, 0.1, 0.5, 1.0, 3.0],
                "random_state": [0, 13, 42]
            }
            self.grid = GridSearchCV(estimator=self.logreg, param_grid=self.param_grid, 
                                     cv=5,  verbose=3, scoring='roc_auc')
            self.grid.fit(train_x, train_y)
            self.C = self.grid.best_params_['C']
            self.random_state = self.grid.best_params_['random_state']

            self.logreg = LogisticRegression(C=self.C, random_state=self.random_state)
            self.logreg.fit(train_x, train_y)
            self.logger.info(f'LogisticRegression best params: {self.grid.best_params_} \
            Exited the get_best_params_for_logreg method of the ModelFinder class')
            return self.logreg
        except Exception as e:
            self.logger.error(f'Exception occured in get_best_params_for_logreg method of the ModelFinder class. \
            Exception message: {e}')
            self.logger.info('LogisticRegression training failed. Exited the get_best_params_for_logreg method of \
            the ModelFinder class')
            raise Exception()

    def get_best_params_for_rf(self, train_x, train_y):
        """Finds the parameters for the RandomForest algorithm which give the best accuracy

           Params:
           train_x: dataframe, training features
           train_y: series, training target (1 if phishing is true and 0 otherwise)

           Returns:
           The model with the best parameters

           Raises:
           Exception in case of failure
        """
        self.logger.info('Entered the get_best_params_for_rf method of the ModelFinder class')

        try:

            self.params_grid = {
                'n_estimators': Integer(120, 1200),
                'max_depth': Integer(5, 30),
                'min_samples_split': Integer(2, 100),
                'min_samples_leaf': Integer(2, 10),
                'max_features': Categorical(['log2', 'sqrt', None]),
                'criterion': Categorical(['gini', 'entropy'])
            }
            
            self.grid = BayesSearchCV(estimator=self.rf, search_spaces=self.params_grid, 
                                      cv=5, n_jobs=-1, n_iter=32, random_state=42, 
                                      verbose=3, scoring='roc_auc')
            self.grid.fit(train_x, train_y)
            self.n_estimators = self.grid.best_params_['n_estimators']
            self.max_depth = self.grid.best_params_['max_depth']
            self.min_samples_split = self.grid.best_params_['min_samples_split']
            self.min_samples_leaf = self.grid.best_params_['min_samples_leaf']
            self.max_features = self.grid.best_params_['max_features']
            self.criterion = self.grid.best_params_['criterion']

            self.rf = RandomForestClassifier(n_estimators=self.n_estimators,
                                             max_depth=self.max_depth,
                                             min_samples_split=self.min_samples_split,
                                             min_samples_leaf=self.min_samples_leaf,
                                             max_features = self.max_features,
                                             criterion = self.criterion,
                                             random_state=42)
            self.rf.fit(train_x, train_y)
            self.logger.info(f'RandomForest best params: {self.grid.best_params_} \
            Exited the get_best_params_for_rf method of the ModelFinder class')
            return self.rf

            # def rf_bayes_func(n_estimators, max_depth, min_samples_split, min_samples_leaf):
            #     """Creates a function of RandomForest parameters to pass to bayesian optimization
            #     """
            #     rf_params = {
            #         'n_estimators': n_estimators,
            #         'max_depth': max_depth,
            #         'min_samples_split': min_samples_split,
            #         'min_samples_leaf': min_samples_leaf
            #     }

            #     score_list = cross_val_score(RandomForestClassifier(random_state=42, **rf_params), 
            #                                  train_x, train_y, cv=5, scoring='roc_auc', n_jobs=-1)
            #     score = sum(score_list) / len(score_list)
            #     return score

            # params_grid = {
            #     'n_estimators': (120, 1200),
            #     'max_depth': (5, 30),
            #     'min_samples_split': (1, 100),
            #     'min_samples_leaf': (1, 10)
            # }

            # rf_bo_result = BayesianOptimization(rf_bayes_func, params_grid, random_state=111)
            # rf_bo_result.maximize(init_points=5, n_iter=100)
            # params_rf = rf_bo_result.max['params']
            # self.n_estimators = round(params_rf['n_estimators'])
            # self.max_depth = round(params_rf['max_depth'])
            # self.min_samples_split = params_rf['min_samples_split']
            # self.min_samples_leaf = params_rf['min_samples_leaf']

            # self.rf = RandomForestClassifier(n_estimators=self.n_estimators,
            #                                  max_depth=self.max_depth,
            #                                  min_samples_split=self.min_samples_split,
            #                                  min_samples_leaf=self.min_samples_leaf,
            #                                  random_state=42)
            # self.rf.fit(train_x, train_y)
            # self.logger.info(f'RandomForest best params: {params_rf} \
            # Exited the get_best_params_for_rf method of the ModelFinder class')
            # return self.rf
        except Exception as e:
            self.logger.error(f'Exception occured in get_best_params_for_rf method of the ModelFinder class. \
            Exception message: {e}')
            self.logger.info('RandomForest training failed. Exited the get_best_params_for_rf method of \
            the ModelFinder class')
            raise Exception()


    def get_best_params_for_lgbm(self, train_x, train_y):
        """Finds the parameters for the LightGBM algorithm which give the best accuracy

           Params:
           train_x: dataframe, training features
           train_y: series, training target (1 if phishing is true and 0 otherwise)

           Returns:
           The model with the best parameters

           Raises:
           Exception in case of failure
        """
        self.logger.info('Entered the get_best_params_for_lgb method of the ModelFinder class')

        try:

            self.params_grid = {
                'num_leaves': Integer(2, 120),
                'min_child_samples': Integer(2, 200),
                'max_bin': Integer(50, 300),
                'subsample': Real(0.4, 0.95),
                'colsample_bytree': Real(0.4, 0.95),
                'min_child_weight': Integer(2, 200),
                'reg_alpha' : Real(1e-6, 2, prior='log-uniform'),
                'reg_lambda' : Real(1e-6, 2, prior='log-uniform'),
                'learning_rate': Real(1e-3, 1, prior='log-uniform')
            }
            
            self.grid = BayesSearchCV(estimator=self.lgbm, search_spaces=self.params_grid, 
                                      cv=5, n_jobs=-1, n_iter=32, random_state=42, verbose=3, 
                                      scoring='roc_auc')
            self.grid.fit(train_x, train_y)
            self.num_leaves = self.grid.best_params_['num_leaves']
            self.min_child_samples = self.grid.best_params_['min_child_samples']
            self.max_bin = self.grid.best_params_['max_bin']
            self.subsample = self.grid.best_params_['subsample']
            self.colsample_bytree = self.grid.best_params_['colsample_bytree']
            self.min_child_weight = self.grid.best_params_['min_child_weight']
            self.reg_alpha = self.grid.best_params_['reg_alpha']
            self.reg_lambda = self.grid.best_params_['reg_lambda']
            self.learning_rate = self.grid.best_params_['learning_rate']

            self.lgbm = LGBMClassifier(
                            boosting_type='dart',
                            objective='binary',
                            metric='binary_logloss',
                            max_depth=-1,
                            subsample_freq=0,
                            min_split_gain=0,
                            random_state=42,
                            num_leaves=self.num_leaves,
                            min_child_samples=self.min_child_samples,
                            max_bin=self.max_bin,
                            subsample=self.subsample,
                            colsample_bytree=self.colsample_bytree,
                            min_child_weight=self.min_child_weight,
                            reg_alpha=self.reg_alpha,
                            reg_lambda=self.reg_lambda,
                            learning_rate=self.learning_rate
                        )
            self.lgbm.fit(train_x, train_y)
            self.logger.info(f'LightGBM best params: {self.grid.best_params_} \
            Exited the get_best_params_for_lgbm method of the ModelFinder class')
            return self.lgbm

            # def lgb_bayes_func(num_leaves, min_child_samples, max_bin, subsample, 
            #                    colsample_bytree, min_child_weight, reg_alpha, reg_lambda):
            #     """Creates a function of LightGBM parameters to pass to bayesian optimization
            #     """
            #     lgb_params = {
            #         'boosting_type': 'dart',
            #         'objective': 'binary',
            #         'metric':'binary_logloss',
            #         'learning_rate': 0.3,
            #         'num_leaves': int(num_leaves),  # we should let it be smaller than 2^(max_depth)
            #         'max_depth': -1,  # -1 means no limit
            #         'min_child_samples': int(min_child_samples),  # Minimum number of data need in a child(min_data_in_leaf)
            #         'max_bin': int(max_bin),  # Number of bucketed bin for feature values
            #         'subsample': subsample,  # Subsample ratio of the training instance.
            #         'subsample_freq': 0,  # frequence of subsample, <=0 means no enable
            #         'colsample_bytree': colsample_bytree,  # Subsample ratio of columns when constructing each tree.
            #         'min_child_weight': int(min_child_weight),  # Minimum sum of instance weight(hessian) needed in a child(leaf)
            #         'min_split_gain': 0,  # lambda_l1, lambda_l2 and min_gain_to_split to regularization
            #         'reg_alpha': reg_alpha,  # L1 regularization term on weights
            #         'reg_lambda': reg_lambda,  # L2 regularization term on weights
            #         'verbose': 0,
            #         'nthread': 32,
            #         'num_class': 10
            #     }

            #     score_list = cross_val_score(LGBMClassifier(**lgb_params), 
            #                                  train_x, train_y, cv=5, scoring='roc_auc', n_jobs=-1)
            #     score = sum(score_list) / len(score_list)
            #     return score

            # params_grid = {
            #     'num_leaves': (2, 120),
            #     'min_child_samples': (1, 200),
            #     'max_bin': (50, 300),
            #     'subsample': (0.4, 0.95),
            #     'colsample_bytree': (0.4, 0.95),
            #     'min_child_weight': (0.0, 200),
            #     'reg_alpha' : (0., 2.),
            #     'reg_lambda' : (0., 2.)
            # }

            # lgb_bo_result = BayesianOptimization(lgb_bayes_func, params_grid, random_state=111)
            # lgb_bo_result.maximize(init_points=45, n_iter=20)
            # params_lgb = lgb_bo_result.max['params']
            # self.num_leaves = round(params_lgb['num_leaves'])
            # self.min_child_samples = round(params_lgb['min_child_samples'])
            # self.max_bin = round(params_lgb['max_bin'])
            # self.subsample = params_lgb['subsample']
            # self.colsample_bytree = params_lgb['colsample_bytree']
            # self.min_child_weight = round(params_lgb['min_child_weight'])
            # self.reg_alpha = params_lgb['reg_alpha']
            # self.reg_lambda = params_lgb['reg_lambda']

            # self.lgb = LGBMClassifier(boosting_type='dart',
            #                           objective ='binary',
            #                           metric='binary_logloss',
            #                           learning_rate=0.3,
            #                           num_leaves=self.num_leaves,
            #                           max_depth=-1,
            #                           min_child_samples=self.min_child_samples,
            #                           max_bin=self.max_bin,
            #                           subsample=self.subsample,
            #                           subsample_freq=0,
            #                           colsample_bytree=self.colsample_bytree,
            #                           min_child_weight=self.min_child_weight,
            #                           min_split_gain=0,
            #                           reg_alpha=self.reg_alpha,
            #                           reg_lambda=self.reg_lambda,
            #                           verbose=0,
            #                           nthread=32,
            #                           num_class=10)
            # self.lgb.fit(train_x, train_y)
            # self.logger.info(f'LightGBM best params: {params_lgb} \
            # Exited the get_best_params_for_lgb method of the ModelFinder class')
            # return self.lgb
        except Exception as e:
            self.logger.error(f'Exception occured in get_best_params_for_lgb method of the ModelFinder class. \
            Exception message: {e}')
            self.logger.info('LightGBM training failed. Exited the get_best_params_for_lgb method of \
            the ModelFinder class')
            raise Exception()

    # def get_best_params_for_cb(self, train_x, train_y):
    #     """Finds the parameters for the CatBoost algorithm which give the best accuracy
    #
    #        Params:
    #        train_x: dataframe, training features
    #        train_y: series, training target (1 if phishing is true and 0 otherwise)
    #
    #        Returns:
    #        The model with the best parameters
    #
    #        Raises:
    #        Exception in case of failure
    #     """
    #     self.logger.info('Entered the get_best_params_for_cb method of the ModelFinder class')
    #
    #     try:
    #
    #         self.params_grid = {
    #             'max_depth': Integer(2, 16),
    #             'l2_leaf_reg': Integer(2, 30),
    #             'iterations': Integer(2, 200)
    #         }
    #
    #         self.grid = BayesSearchCV(estimator=self.cb, search_spaces=self.params_grid,
    #                                   cv=5, n_iter=32, random_state=42, verbose=3,
    #                                   scoring='roc_auc')
    #         self.grid.fit(train_x, train_y)
    #         self.max_depth = self.grid.best_params_['max_depth']
    #         self.l2_leaf_reg = self.grid.best_params_['l2_leaf_reg']
    #         self.iterations = self.grid.best_params_['iterations']
    #
    #         self.cb = CatBoostClassifier(
    #                       max_depth=self.max_depth,
    #                       l2_leaf_reg=self.l2_leaf_reg,
    #                       iterations=self.iterations
    #                   )
    #         self.cb.fit(train_x, train_y)
    #         self.logger.info(f'CatBoost best params: {self.grid.best_params_} \
    #         Exited the get_best_params_for_cb method of the ModelFinder class')
    #         return self.cb
    #
    #         # def cb_bayes_func(n_estimators):
    #         #     """Creates a function of CatBoost parameters to pass to bayesian optimization
    #         #     """
    #         #     cb_params = {
    #         #         'n_estimators': n_estimators,
    #         #     }
    #
    #         #     score_list = cross_val_score(CatBoostClassifier(**cb_params),
    #         #                                  train_x, train_y, cv=5, scoring='roc_auc', n_jobs=-1)
    #         #     score = sum(score_list) / len(score_list)
    #         #     return score
    #
    #         # params_grid = {
    #         #     'n_estimators': (10, 300)
    #         # }
    #
    #         # cb_bo_result = BayesianOptimization(cb_bayes_func, params_grid, random_state=111)
    #         # cb_bo_result.maximize(init_points=45, n_iter=20)
    #         # params_cb = cb_bo_result.max['params']
    #         # self.n_estimators = round(params_cb['n_estimators'])
    #
    #         # self.cb = CatBoostClassifier(n_estimators=self.n_estimators)
    #         # self.cb.fit(train_x, train_y)
    #         # self.logger.info(f'CatBoost best params: {params_cb} \
    #         # Exited the get_best_params_for_cb method of the ModelFinder class')
    #         # return self.cb
    #     except Exception as e:
    #         self.logger.error(f'Exception occured in get_best_params_for_cb method of the ModelFinder class. \
    #         Exception message: {e}')
    #         self.logger.info('CatBoost training failed. Exited the get_best_params_for_cb method of \
    #         the ModelFinder class')
    #         raise Exception()

    def get_best_model(self, train_x, train_y, test_x, test_y):
        """Finds out a model which has the best ROC_AUC score

           Params:
           train_x: dataframe, training features
           train_y: series, training target (1 if phishing is true and 0 otherwise)
           test_x: dataframe, testing features
           test_y: series, testing target (1 if phishing is true and 0 otherwise)

           Returns:
           The best model name and the model object

           Raises:
           Exception in case of failure
        """
        self.logger.info('Entered the get_best_model method of the ModelFinder class')

        try:
            # create best model for XGBoost
            self.xgboost = self.get_best_params_for_xgboost(train_x, train_y)
            self.prediction_xgboost = self.xgboost.predict(test_x)
            # If there is only one label in y, then roc_auc_score returns error.
            # We will use accuracy in that case.
            if len(test_y.unique()) == 1:
                self.xgboost_score = accuracy_score(test_y, self.prediction_xgboost)
                self.logger.info(f'Accuracy for XGBoost: {self.xgboost_score}')
            else:
                self.xgboost_score = roc_auc_score(test_y, self.prediction_xgboost)
                self.logger.info(f'ROC-AUC for XGBoost: {self.xgboost_score}')

            # create best model for SVC
            self.svm = self.get_best_params_for_svm(train_x, train_y)
            self.prediction_svm = self.svm.predict(test_x)
            # If there is only one label in y, then roc_auc_score returns error.
            # We will use accuracy in that case
            if len(test_y.unique()) == 1:
                self.svm_score = accuracy_score(test_y, self.prediction_svm)
                self.logger.info(f'Accuracy for SVM: {self.svm_score}')
            else:
                self.svm_score = roc_auc_score(test_y, self.prediction_svm)
                self.logger.info(f'ROC-AUC for SVM: {self.svm_score}')

            # create best model for LogisticRegression
            self.lr = self.get_best_params_for_logreg(train_x, train_y)
            self.prediction_lr = self.lr.predict(test_x)
            # If there is only one label in y, then roc_auc_score returns error.
            # We will use accuracy in that case
            if len(test_y.unique()) == 1:
                self.lr_score = accuracy_score(test_y, self.prediction_lr)
                self.logger.info(f'Accuracy for LogisticRegression: {self.lr_score}')
            else:
                self.lr_score = roc_auc_score(test_y, self.prediction_lr)
                self.logger.info(f'ROC-AUC for LogisticRegression: {self.lr_score}')

            # create best model for StackedNaiveBayes
            # self.gnb = GaussianNB()
            # cont_col_list = [col for col in train_x.columns if train_x[col].nunique()>2]
            # cat_col_list = [col for col in train_x.columns if col not in cont_col_list]
            # self.gnb.fit(train_x[cont_col_list], train_y)
            #
            # self.bnb = BernoulliNB()
            # self.gnb.fit(train_x[cat_col_list], train_y)
            #
            # gnb_train = pd.DataFrame({
            #                           'gnb': self.gnb.predict_proba(train_x[cont_col_list])[:, 1],
            #                           'bnb': self.bnb.predict_proba(train_x[cat_col_list])[:, 1],
            #             })
            # gnb_test = pd.DataFrame({
            #                           'gnb': self.gnb.predict_proba(test_x[cont_col_list])[:, 1],
            #                           'bnb': self.bnb.predict_proba(test_x[cat_col_list])[:, 1],
            #             })
            #
            # self.nb = GaussianNB()
            # self.nb.fit(gnb_train, train_y)
            # self.prediction_nb = self.nb.predict(gnb_test)
            # # If there is only one label in y, then roc_auc_score returns error.
            # # We will use accuracy in that case
            # if len(test_y.unique()) == 1:
            #     self.nb_score = accuracy_score(test_y, self.prediction_nb)
            #     self.logger.info(f'Accuracy for StackedNaiveBayes: {self.nb_score}')
            # else:
            #     self.nb_score = roc_auc_score(test_y, self.prediction_nb)
            #     self.logger.info(f'ROC-AUC for StackedNaiveBayes: {self.nb_score}')

            # create best model for MixedNaiveBayes
            col_list = list(train_x.columns)
            cat_index_list = [col_list.index(col) for col in col_list if train_x[col].nunique()<2]
            self.mnb = MixedNB(categorical_features=cat_index_list)
            self.mnb.fit(train_x, train_y)
            self.prediction_mnb = self.mnb.predict(test_x)
            # If there is only one label in y, then roc_auc_score returns error.
            # We will use accuracy in that case
            if len(test_y.unique()) == 1:
                self.mnb_score = accuracy_score(test_y, self.prediction_mnb)
                self.logger.info(f'Accuracy for MixedNaiveBayes: {self.mnb_score}')
            else:
                self.mnb_score = roc_auc_score(test_y, self.prediction_mnb)
                self.logger.info(f'ROC-AUC for MixedNaiveBayes: {self.mnb_score}')

            # create best model for RandomForest
            self.rf = self.get_best_params_for_rf(train_x, train_y)
            self.prediction_rf = self.rf.predict(test_x)
            # If there is only one label in y, then roc_auc_score returns error.
            # We will use accuracy in that case
            if len(test_y.unique()) == 1:
                self.rf_score = accuracy_score(test_y, self.prediction_rf)
                self.logger.info(f'Accuracy for RandomForest: {self.rf_score}')
            else:
                self.rf_score = roc_auc_score(test_y, self.prediction_rf)
                self.logger.info(f'ROC-AUC for RandomForest: {self.rf_score}')

            # create best model for LightGBM
            self.lgbm = self.get_best_params_for_lgbm(train_x, train_y)
            self.prediction_lgbm = self.lgbm.predict(test_x)
            # If there is only one label in y, then roc_auc_score returns error.
            # We will use accuracy in that case
            if len(test_y.unique()) == 1:
                self.lgbm_score = accuracy_score(test_y, self.prediction_lgbm)
                self.logger.info(f'Accuracy for LightGBM: {self.lgbm_score}')
            else:
                self.lgbm_score = roc_auc_score(test_y, self.prediction_lgbm)
                self.logger.info(f'ROC-AUC for LightGBM: {self.lgbm_score}')

            # create best model for CatBoost
            # self.cb = self.get_best_params_for_cb(train_x, train_y)
            # self.prediction_cb = self.cb.predict(test_x)
            # # If there is only one label in y, then roc_auc_score returns error.
            # # We will use accuracy in that case
            # if len(test_y.unique()) == 1:
            #     self.cb_score = accuracy_score(test_y, self.prediction_cb)
            #     self.logger.info(f'Accuracy for CatBoost: {self.cb_score}')
            # else:
            #     self.cb_score = roc_auc_score(test_y, self.prediction_cb)
            #     self.logger.info(f'ROC-AUC for CatBoost: {self.cb_score}')

            # comparing the models
            score_list = [self.svm_score, self.xgboost_score, self.lr_score,
                          self.mnb_score, self.rf_score, self.lgbm_score]
            model_name_list = ['SVM', 'XGBoost', 'LogisticRegression',
                               'MixedNaiveBayes', 'RandomForest', 'LightGBM']
            model_list = [self.svm, self.xgboost, self.lr,
                          self.mnb, self.rf, self.lgbm]

            max_score = 0
            for i_score in score_list:
                if i_score > max_score:
                    max_score = i_score
                else:
                    pass

            return model_name_list[score_list.index(max_score)], model_list[score_list.index(max_score)]

        except Exception as e:
            self.logger.error(f'Exception occurred in get_best_model method of the ModelFinder class. \
            Exception message: {e}')
            self.logger.info('Model selection Failed. Exited the get_best_model method of the Model_Finder class')
            raise Exception()

