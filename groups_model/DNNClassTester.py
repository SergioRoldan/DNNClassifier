import groupDNNClassifier as DNNC
import os
from datetime import datetime
import time

start = time.clock()

now = str(datetime.now()).split('.')[0].replace(" ", "_").replace(":", "_").replace("-", "_")
directory = "runs/" + now

learning_rate = 0.000013

if not os.path.exists(directory):
    os.makedirs(directory)

for i in range(1, 2):
    partial_start = time.clock()

    print("Learning Rate: ",learning_rate*i)

    DNNC.run(learning_rate*i, 2000, 20, 20, [10, 10, 10, 10], directory + '/')

    partial_end = time.clock()
    print("Iteration ", i, " time : ", (partial_end-partial_start)/60)   

end = time.clock()
print("Total execution time: ",(end-start)/60) 
