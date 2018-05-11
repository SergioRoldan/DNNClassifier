import symbolDNNClassifier as DNNC
import os
from datetime import datetime
import time
import gc

start = time.clock()

now = str(datetime.now()).split('.')[0].replace(' ', '_').replace(':', '_').replace('-', '_')
directory = 'runs/' + now

learning_rate = 0.000858
hidden_layers = [
        10,
        10,
        10,
        10
]
batch_size = 40
steps = 2500

if not os.path.exists(directory):
    os.makedirs(directory)

for i in range(1, 5):
    
    partial_start = time.clock()
    print('Learning Rate: ',learning_rate+i*10**(-6))

    DNNC.run(learning_rate+i*10**(-6), steps, batch_size, 15, hidden_layers, directory + '/')
    
    gc.collect()

    partial_end = time.clock()
    print('Iteration ', i, ' time : ', (partial_end-partial_start)/60)   

end = time.clock()
print('Total execution time: ',(end-start)/60) 

'''
https://10.110.8.42:8080
'''