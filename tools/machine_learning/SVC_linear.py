import time

from sklearn import svm

from data_holder import data_holder


class SVC_linear(data_holder):
    def __init__(self, CC=1, gamma=0.7, degree=3):
        self.algo_name = "SVC_linear"
        
        indv_start = time.time()
        self.ml_algo = svm.LinearSVC(
            dual=False,
            tol=0.00001,
            C=1.0,
            multi_class='ovr',
            fit_intercept=True,
            intercept_scaling=1,
            class_weight={0: 9},
            verbose=0,
            random_state=None,
            max_iter=1000000
        )
        self.fit()
        indv_end = time.time()
        
        self.output(" Initialized in " + str(indv_end - indv_start) + " sec")
        
        
