import time
from sklearn.linear_model import PassiveAggressiveClassifier
from data_holder import data_holder
from sklearn import linear_model


class Pas_agg_class(data_holder):
    def __init__(self, CC=1, gamma=0.7, degree=3):
        self.algo_name = "Passive Aggressive Classifier"
        
        indv_start = time.time()
        
        self.ml_algo = linear_model.PassiveAggressiveClassifier(
            C=1.0,
            fit_intercept=True,
            n_iter=5,
            shuffle=True,
            verbose=0,
            loss='hinge',
            n_jobs=1,
            random_state=None,
            warm_start=False,
            class_weight={1:10}
        )
        
        self.fit()
        indv_end = time.time()
        
        self.output(" Initialized in " + str(indv_end - indv_start) + " sec")
        
        
