import random
import time
import os.path, sys
from tools.db import db
from tools.machine_learning.SGD import SGD
from tools.machine_learning.SVC_dummy import SVC_dummy
from tools.machine_learning.SVC_linear import SVC_linear
from tools.machine_learning.SVC_poly import SVC_poly
from tools.machine_learning.SVC_rbf import SVC_rbf
from tools.machine_learning.SVC_sigmoid import SVC_sigmoid
from tools.machine_learning.data_holder import data_holder
from tools.machine_learning.Bayes_Ridge_Reg import Bayes_Ridge_Reg
from tools.machine_learning.Logistic_Reg import Logistic_Reg
from tools.machine_learning.ARD_Reg import ARD_Reg
from tools.machine_learning.Perceptron import Perceptron
from tools.machine_learning.Pas_agg_class import Pas_agg_class
from tools.machine_learning.KNeighborsClassifier import KNeighborsClassifier
from tools.machine_learning.Pipe_LinearSVC_KBest import Pipe_LinearSVC_KBest

# from tools import machinelearning as ml
def output(msg):
    for line in msg.splitlines():
        print "Main: " + line
        
        
def check_db_data(tmp_tuple, db_data):
    if(len(tmp_tuple) != len(db_data)):
        output("Length not equal!\nGenerated data length " + str(len(tmp_tuple)) + "\nDB data " + str(len(db_data)))
        raise Exception("Length not equal!")
        return
        
    if(len(tmp_tuple) == 0):
        raise Exception("Tuple list empty!")
        return
    
    if(len(db_data) == 0):
        raise Exception("DB list empty!")
        return
    
    for i in range(len(tmp_tuple)):
        tmp = tmp_tuple[i]
        for j in range(len(tmp)):
            if(db_output[i][j] != tmp[j]):
                output("Mismatch! \n DB:" + str(db_output[i][j]) + "\n but should be " + str(tmp[j]) + "\n")
            
    output("Lists are equivalent")
    
    
#
# Start of main program
#                

my_db = db("test_db.db",
           ["Period1", "Period2", "Deadline_Reached"],
           ["int", "int", "bool"])
# ["ID","Priority","Period","ID2","Deadline_Reached"],
# ["int", "int", "int", "int", "bool"])


# total number of sets
num_sets1 = 30
# num_sets2 = 50
num_trainingsets = num_sets1  # + num_sets2
data_per_set = len(my_db.attr)


tmp_tuple = []

start = time.time()


# change to 2 for more test data
num_tests = 1

# fill tuple data
for i in range(int(num_sets1 * 0.5 * num_tests)):
    for j in range(int(num_sets1 * 0.5 * num_tests)):
        period1 = 200 + i * 20 / num_tests + int(random.random() * 50)
        period2 = 240 + j * 20 / num_tests + int(random.random() * 50) 
        
        tmp_tuple.append((period1, period2, period1 * i * period1 + period2 * period2 * j < 2500000.0 * num_tests + random.random() * 500000))
        # tmp_tuple.append((period1, period2, period1 + period2  <  750.0 * num_tests + random.random() * 10))

end = time.time()
output("Test data generated! Total number: "+str(num_sets1*num_sets1)+" Time needed: " + str(end - start))
    


# Write tmp_data to db
output("Write tmp data to database!")
my_db.write(tmp_tuple)

db_output = []

# Read values from db
output("Read data from database!")
# db_output = my_db.read()

# check_db_data(tmp_tuple, db_output)


# Machine Learning and output

# Get data from database 
output_data = my_db.read_output()
input_data = my_db.read_input()

output("Create machinelearning")


# old stuff
# machine_learning = ml.machinelearning(data_per_set, num_trainingsets)
# machine_learning.training_phase(input_data, output_data)
# machine_learning.save_training("ml")
# machine_learning.load_training("ml")
# machine_learning.plot(h=5)    

data_holder = data_holder(data_per_set, num_trainingsets, input_data, output_data)       

# create machine learning algorithmns
svc_linear = SVC_linear()
svc_poly = SVC_poly()
svc_rbf = SVC_rbf()
#svc_dummy = SVC_dummy()
svc_sigmoid = SVC_sigmoid()
sgd = SGD()
bay_reg = Bayes_Ridge_Reg()
log_reg = Logistic_Reg()
#ard_reg = ARD_Reg()
percep = Perceptron()
pas_agg_class = Pas_agg_class()
k_neigh = KNeighborsClassifier()


#pipe = Pipe_LinearSVC_KBest()



# Create list
data_holder.plot([svc_linear, svc_poly, svc_rbf,svc_sigmoid,sgd, bay_reg, log_reg, percep, pas_agg_class, k_neigh], 3, 1)


