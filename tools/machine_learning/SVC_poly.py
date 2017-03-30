import time

from sklearn import svm

from data_holder import data_holder


class SVC_poly(data_holder):
    def __init__(self):
        self.algo_name = "SVC_poly"
                
        indv_start = time.time()
        self.ml_algo = svm.SVC(
            kernel='poly',
            degree=2,
            cache_size=7000,
            tol=0.01,
            gamma='auto',
            coef0=0.0,
            shrinking=True,
            class_weight={1: 20},
            C=1,
            probability=False,
            verbose=False,
            max_iter=-1,
            decision_function_shape=None,
            random_state=None
        )
        
        self.fit()
        indv_end = time.time()
        
        self.output(" Initialized in " + str(indv_end - indv_start) + " sec")        
