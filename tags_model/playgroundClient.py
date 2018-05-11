import zmq
import random
import time
import json
import numpy as np
import ast

#Mode data parsing to server
start_con = time.clock()

port = '7999'
context = zmq.Context()
print("Client started")
socket = context.socket(zmq.REQ)
socket.connect("tcp://localhost:%s" %port)

client_id = str(random.randint(0,10)).encode('utf-8') 

print("Sending request to server")
socket.send(b"Request from client %s" % client_id)
message = socket.recv()
print("Received reply: %s" % message)

new_port = message.decode('utf-8').split(': ')[1]

socket.close()

finish_con = time.clock()
start_work = time.clock()

socket = context.socket(zmq.REQ)
socket.connect('tcp://localhost:%s' % new_port)


print("Sending ping to worker")
socket.send(b'Ping from client %s' % client_id)
message = socket.recv()
print('Received reply: %s' % message)

finish_work = time.clock()
start_req = time.clock()

print('Sending request to worker')
someObj = {
    'foo': 'foo',
    'bar': 'bar'
}
socket.send(('Inference from client %s: %s' % (client_id, json.dumps(someObj))).encode('utf-8'))
message = socket.recv().decode('utf-8')
print('Received reply: %s' % message.split(': ')[0])

inferences = message.split(': ')[1].replace('[','').replace(']','')
inferences = inferences.split('), (')

inferences = [('('+inference) for inference in inferences if '(' not in inference]
inferences = [(inference+')') for inference in inferences if ')' not in inference]
inferences = [eval(item) for item in inferences]

finish_req = time.clock()

'''for inferece in inferences: print(inferece)'''
# TODO // Probs refence to each of 15 cathegories, change the function in server and here

socket.send(b'Ack & close from client %s' % client_id)

time.sleep(0.5)
context.destroy()

print('Time till a new port is asigned: ', finish_con-start_con)
print('Time till the worker connection is tested: ', finish_work-start_work)
print('Time till a request in answer and parsed: ', finish_req-start_req)


print('Client finished')