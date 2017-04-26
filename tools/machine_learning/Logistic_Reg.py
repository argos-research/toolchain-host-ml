import time
from sklearn import linear_model
from data_holder import data_holder


class Logistic_Reg(data_holder):
    def __init__(self, CC=1, gamma=0.7, degree=3):
        self.algo_name = "Logistic_Reg"
        
        indv_start = time.time()
        self.ml_algo = linear_model.LogisticRegression(
            penalty='l2',
            dual=False,
            tol=0.00000001,
            C=1.0,
            fit_intercept=True,
            intercept_scaling=5,
            class_weight={1: 2},
            random_state=None,
            solver='liblinear',
            max_iter=100,
            multi_class='ovr',
            verbose=0,
            warm_start=True,
            n_jobs=1
        )
        
        self.fit()
        indv_end = time.time()
        
        self.output(" Initialized in " + str(indv_end - indv_start) + " sec")
        
        
