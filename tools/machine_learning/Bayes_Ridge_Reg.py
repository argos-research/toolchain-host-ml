import time
from sklearn import linear_model
from data_holder import data_holder


class Bayes_Ridge_Reg(data_holder):
    def __init__(self, CC=1, gamma=0.7, degree=3):
        self.algo_name = "Bayes_Ridge_Reg"
        
        indv_start = time.time()
        self.ml_algo = linear_model.BayesianRidge(
            n_iter=300,
            tol=0.0000000001,
            alpha_1=1e-06,
            alpha_2=1e-06,
            lambda_1=1e-06,
            lambda_2=1e-06,
            compute_score=True,
            fit_intercept=True,
            normalize=True,
            copy_X=False,
            verbose=False
       )
        self.fit()
        indv_end = time.time()
        
        self.output(" Initialized in " + str(indv_end - indv_start) + " sec")
        
        
