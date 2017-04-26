import time

from sklearn.neighbors import KNeighborsRegressor

from data_holder import data_holder


class KNeighborsClassifier(data_holder):
    def __init__(self):
        self.algo_name = "KNeighborsClassifier"
        
        indv_start = time.time()
        self.ml_algo = KNeighborsRegressor(
            n_neighbors=1,
            weights='distance',
            algorithm='auto',
            leaf_size=80,
            p=2,
            metric='minkowski',
            metric_params=None,
            n_jobs=-1
        )
        self.fit()
        indv_end = time.time()
        
        self.output(" Initialized in " + str(indv_end - indv_start) + " sec")
        
        
