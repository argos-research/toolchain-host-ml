import time
from sklearn.linear_model import ARDRegression
from data_holder import data_holder


class ARD_Reg(data_holder):
    def __init__(self, CC=1, gamma=0.7, degree=3):
        self.algo_name = "ARD_Reg"
        
        indv_start = time.time()
        
        self.ml_algo = ARDRegression(
            compute_score=True
        )
        
        self.fit()
        indv_end = time.time()
        
        self.output(" Initialized in " + str(indv_end - indv_start) + " sec")
        
        
