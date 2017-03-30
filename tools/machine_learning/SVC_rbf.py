import time

from sklearn import svm

from data_holder import data_holder


class SVC_rbf(data_holder):
    def __init__(self, CC=1, gamma=0.7, degree=3):
        self.algo_name = "SVC_rbf"
        
        indv_start = time.time()
        self.ml_algo = svm.SVC(kernel='rbf', cache_size=7000, tol=0.00001, class_weight={0: 20}, gamma=0.01, C=1)
        self.fit()
        indv_end = time.time()
        
        self.output(" Initialized in " + str(indv_end - indv_start) + " sec")
