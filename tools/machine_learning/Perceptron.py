import time
from sklearn.linear_model import Perceptron
from data_holder import data_holder
from sklearn import linear_model


class Perceptron(data_holder):
    def __init__(self, CC=1, gamma=0.7, degree=3):
        self.algo_name = "Perceptron"
        
        indv_start = time.time()
        
        self.ml_algo = linear_model.Perceptron(
            penalty="l1",
            alpha=0.1,
            fit_intercept=True,
            n_iter=10,
            shuffle=True,
            verbose=0,
            eta0=1.0,
            n_jobs=1,
            random_state=1,
            class_weight={0:1},
            warm_start=False
        )
        
        self.fit()
        indv_end = time.time()
        
        self.output(" Initialized in " + str(indv_end - indv_start) + " sec")
        
        
