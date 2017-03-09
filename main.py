from tools.db import db
from tools import machinelearning as ml
import time
import random

def output(msg):
    for line in msg.splitlines():
        print "Main: "+line
        
        
def check_db_data(tmp_tuple, db_data):
    if(len(tmp_tuple)!=len(db_data)):
        output("Length not equal!\nGenerated data length "+str(len(tmp_tuple))+"\nDB data "+str(len(db_data)))
        raise Exception("Length not equal!")
        return
        
    if(len(tmp_tuple)==0):
        raise Exception("Tuple list empty!")
        return
    
    if(len(db_data)==0):
        raise Exception("DB list empty!")
        return
    
    for i in range(len(tmp_tuple)):
        tmp = tmp_tuple[i]
        for j in range(len(tmp)):
            if(db_output[i][j]!=tmp[j]):
                output("Mismatch! \n DB:"+str(db_output[i][j])+"\n but should be "+str(tmp[j])+"\n")
            
    output("Lists are equivalent")
    
    
#
# Start of main program
#                

my_db = db("test_db.db", 
           ["Period1","Period2","Deadline_Reached"],
           ["int", "int", "bool"])
#["ID","Priority","Period","ID2","Deadline_Reached"],
#["int", "int", "int", "int", "bool"])


#total number of sets
num_sets1 = 30
#num_sets2 = 50
num_trainingsets = num_sets1 #+ num_sets2
data_per_set = len(my_db.attr)


tmp_tuple = []

start = time.time()


# change to 2 for more test data
num_tests = 1

#fill tuple data
for i in range(int(num_sets1*0.5*num_tests)):
    for j in range(int(num_sets1*0.5*num_tests)):
        period1 = 200+i*20/num_tests + int(random.random()*10)
        period2 = 240+j*20/num_tests + int(random.random()*10) 
        
        tmp_tuple.append((period1, period2, period1 * i * period1 + period2 * period2 * j  <  2500000.0 * num_tests + random.random() * 500000))
        #tmp_tuple.append((period1, period2, period1 + period2  <  750.0 * num_tests + random.random() * 10))

end = time.time()
output("Test data generated! Time needed: "+str(end-start))
    


#Write tmp_data to db
output("Write tmp data to database!")
my_db.write(tmp_tuple)

db_output = []

#Read values from db
output("Read data from database!")
#db_output = my_db.read()

#check_db_data(tmp_tuple, db_output)


#Machine Learning and output

#Get data from database 
output_data = my_db.read_output()
input_data = my_db.read_input()

output("Create machinelearning")


machine_learning = ml.machinelearning(data_per_set, num_trainingsets)
machine_learning.training_phase(input_data, output_data)

machine_learning.plot(h=5)            


