import time
from data_holder import data_holder
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest
from sklearn import svm
from sklearn.feature_selection import f_regression

class Pipe_LinearSVC_KBest(data_holder):
    def __init__(self, CC=1, gamma=0.7, degree=3):
        self.algo_name = "Pipe_LinearSVC_Neigh"
        
        indv_start = time.time()
        
        
        KNeigh = SelectKBest(f_regression, k=5)
        linear = svm.SVC(kernel='linear')
        self.ml_algo = Pipeline([('KNeigh', KNeigh), ('linear', linear)])
        
        
        self.ml_algo.set_params(KNeigh__k=1, linear__C=1.0).fit(data_holder.input_2dvector,  data_holder.output_1dvector)
        indv_end = time.time()
        
        self.output(" Initialized in " + str(indv_end - indv_start) + " sec")
        
        
