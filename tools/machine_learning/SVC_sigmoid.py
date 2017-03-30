import time

from sklearn import svm

from data_holder import data_holder


class SVC_sigmoid(data_holder):
    def __init__(self):
        self.algo_name = "SVC_sigmoid"
                
        indv_start = time.time()
        self.ml_algo = svm.SVC(
            kernel='sigmoid',
            degree=1,
            cache_size=7000,
            tol=0.01,
            gamma='auto',
            coef0=0.01,
            shrinking=True,
            class_weight={1: 20},
            C=2,
            probability=False,
            verbose=False,
            max_iter=-1,
            decision_function_shape=None,
            random_state=None
        )
        
        self.fit()
        indv_end = time.time()
        
        self.output(" Initialized in " + str(indv_end - indv_start) + " sec")        
