import base64
import json
import os
import random
import re
import sys
import threading
import time
from datetime import datetime

import numpy as np
import tensorflow as tf
import zmq

import dataHandler as dh
import symbolDNNClassifier as DNNC

# Model attributes for prediction

classes = 15
tags = ['button', 'input', 'textarea', 'alert', 'table', 'footer', 'link',
        'sidebar', 'status', 'paragraph', 'br', 'timeline', 'item', 'header', 'undefined']

'''
Input: One-hot 2-dim list with predictions (e.g. [ [0 1 0 ...] [1 0 0 ...] [0 0 ... 1]])
Returns: Categorical list matching tags (e.g. [button, input, ...]) 
'''


def oneHotToCategorical(onehot):
    numerical = [index for row in onehot for index,
                 _ in enumerate(row) if row[index] == 1.]
    categorical = [tags[int(item)] for item in numerical]

    return categorical


'''
Input: Probabilities 2-dim list for each of the classes (e.g. [ [0.92 0.01 ...] [0.02 0.21 ...] ])
Returns: Higher probability list, confidence % (e.g. [76 94 87 ...])
'''


def handleProbabilities(decProbs):
    probs = []

    for row in decProbs:
        higher = 0
        for col in row:
            if col > higher:
                higher = col

        probs.append(higher * 100)

    return probs


'''
Obtains a random port in the range 8000-9000 and checks if the port is in use
Returns: The new port or 0 if it's already in use
'''


def getNewFreePort():

    sck = context.socket(zmq.REP)
    prt = random.randint(8000, 9000)

    try:
        sck.bind("tcp://*:%s" % prt)
        return prt

    except zmq.ZMQError:
        print("Port already in use")
        return 0


'''
_bytes_feature requires a list
_float_feature & _int_feature require a numeric value
Returns: TensorFlow tfrecord compliant feature
'''


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))


def _float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def _int_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


'''
Input: Name of the client making the request (str), Objects to be infer (list of dicts)
Returns: Path where the tfrecord of objects has been created
- Parse objects as TensorFlow features and write a tfrecord to feed the estimator
'''


def createDataset(client, objs):

    objs = dh.handleInferences(objs)

    client = dh._replaceMultiple(client, [':', '.'], '_')

    filename = 'predictions/' + client + '_'
    filename += dh._replaceMultiple(str(datetime.now()
                                        ).split('.')[0], [' ', ':', '-'], '_')
    filename += '.tfrecord'

    with open(filename, 'w') as text_file:
        print('', file=text_file)

    writer = tf.python_io.TFRecordWriter(filename)

    for index, _ in enumerate(objs):

        feature = {
            'labels':         _int_feature(0),
            'insName':        _bytes_feature(objs[index]['name']),
            'insSymID':       _bytes_feature(objs[index]['symbolID']),
            'insSymIDBool':   _int_feature(0 if b'none' in objs[index]['symbolID'] else 1),
            'insColor':       _bytes_feature(objs[index]['color']),
            'insColorBool':   _int_feature(0 if b'none' in objs[index]['color'] else 1),
            'insText':        _bytes_feature(objs[index]['text']),
            'insFrameClass':  _bytes_feature(objs[index]['frame']['class']),
            'insFrameHeight': _float_feature(float(objs[index]['frame']['height'])),
            'insFrameWidth':  _float_feature(float(objs[index]['frame']['width'])),
            'prevIns':        _bytes_feature(objs[index]['previous']),
            'prevInsBool':    _int_feature(0 if b'none' in objs[index]['previous'] else 1),
            'nxtIns':         _bytes_feature(objs[index]['next']),
            'nxtInsBool':     _int_feature(0 if b'none' in objs[index]['next'] else 1),
            'parent':         _bytes_feature(objs[index]['parent']),
            'parentBool':     _int_feature(0 if b'none' in objs[index]['parent'] else 1),
        }

        for i in range(1, 6):
            obj = 'obj' + str(i)
            feature[obj+'Name'] = _bytes_feature(objs[index][obj]['name'])
            feature[obj+'Class'] = _bytes_feature(objs[index][obj]['class'])
            feature[obj+'Bool'] = _int_feature(
                0 if b'none' in objs[index][obj]['name'] else 1)

        example = tf.train.Example(features=tf.train.Features(feature=feature))
        writer.write(example.SerializeToString())

    writer.close()
    sys.stdout.flush()

    return filename


'''
- Worker thread function
Input: Port to be binded and DNNClassifier to predict
- Worker binds to newPort and waits to receive message from server:
    If the message is inference, the worker makes the prediction and return it
    If the message is ping, the worker pings back
    If the message is close, the worker stops
Note: If any error happen in the socket the worker signal the error and die
'''


def inference(newPort, classifier):
    sock = context.socket(zmq.REP)
    sock.RCVTIMEO = 5000

    try:
        sock.bind("tcp://*:%s" % newPort)
        print("Worker started at port ", newPort)

        while True:
            message = sock.recv().decode('utf-8')
            print("Worker received request")

            if 'inference' in message.lower():

                try:
                    objs = message.split('> ', 1)[1]
                    objs = eval(objs)

                    client = message.split('> ', 1)[0].split('client ', 1)[1]
                    client = dh._replaceMultiple(client, [':', '.'], '_')
                    print(client)

                    dataset_path = createDataset(client, objs)
                    predict_fn = DNNC.create_input_feat_fn([dataset_path],
                                                        len(objs))

                    predictions = list(classifier.predict(input_fn=predict_fn))
                    probabilities = np.array(
                        [item['probabilities'] for item in predictions])
                    pred_class_id = np.array(
                        [item['class_ids'][0] for item in predictions])
                    pred_one_hot = tf.keras.utils.to_categorical(
                        pred_class_id, num_classes=classes)

                    categor = oneHotToCategorical(pred_one_hot)
                    probs = handleProbabilities(probabilities)

                    result = list(zip(categor, probs))

                    print('Worker sending inference results at port', newPort)
                    sock.send((threading.currentThread().getName() +
                            " inference: " + str(result)).encode('utf-8'))
                except Exception as e:
                    print(str(e))
                    sock.send(('Exception <'+str(e)+'> during inference').encode('utf-8'))
                
            elif 'ping' in message.lower():
                print('Worker sending ping at port', newPort)
                sock.send((threading.currentThread().getName() +
                           ' ping').encode('utf-8'))
            elif 'close' in message.lower():
                break

        print('Worker killed at port ', newPort)

    except zmq.ZMQError:
        print("Error & worked killed at port ", newPort)


'''
- Models Serving main: 
    Check and substitue command arguments, if any. Non-defined arguments are ignored
    Train the classifier
    Start a socket to listen node requests:
        If infer request is received, a new port is obtained and a new worker binded to this port and requester is created to serve him
        If metrics request is received, the server returns the metrics of the last training period
'''

if __name__ == '__main__':

    args = sys.argv[1:]

    now = str(datetime.now()).split('.')[0].replace(
        ' ', '_').replace(':', '_').replace('-', '_')

    if '--help' in args or 'help' in args:
        print('Arguments format -> --arg1=val1 --arg2=val2 ...')
        print('--learning-rate=X ; defaults to 0.000862')
        print('--batch-size=X ; defaults to 45')
        print('--steps=X ; defaults to 2000')
        print('-hidden-layers=X ; defaults to 5 hidden layers with 10 node each')
        print('--periods=X ; defaults to 15')
        print('--port=X ; defaults to 7999')
        sys.exit('--directory=X ; defaults to runs/')

    learning_rate =0.018
    batch_size = 45
    steps = 2000
    hidden_layers = [
        10,
        10,
        10,
        10,
        10
    ]
    periods = 30
    directory = 'runs/' + now
    port = 7999

    for arg in args:
        if '--learning-rate' in arg:
            learning_rate = float(arg.split('=')[1])

        if '--batch-size' in arg:
            batch_size = int(arg.split('=')[1])

        if '--steps' in arg:
            steps = int(arg.split('=')[1])

        if '--hidden-layers' in arg:
            hidden_layers = arg.split('=')[1]
            hidden_layers = hidden_layers.split(',')   

        if '--periods' in arg:
            periods = int(arg.split('=')[1])

        if '--directory' in arg:
            directory = arg.split('=')[1] + now          

        if '--port' in arg:
            port = arg.split('=')[1]

    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError as e:
        print(str(e))   
        directory = 'runs/' + now
        pass

    print('Training parameters: ')
    print(' - learning_rate: ', learning_rate)
    print(' - steps: ', steps)
    print(' - mini_batch_size: ', batch_size)
    print(' - periods: ', periods)
    print(' - hidden_layers', hidden_layers)
    print(' - Optimizer: Nesterov Momentum with lambda = 0.9 and clipping = 3')

    classifier, metrics_dir = DNNC.train_and_evaluate(
        learning_rate, steps, batch_size, periods, hidden_layers, directory + '/')

    context = zmq.Context()
    socket = context.socket(zmq.REP)
    socket.bind("tcp://*:%s" % port)

    print("Model Serving started")

    while True:
        message = socket.recv().decode('utf-8')
        print("Server Received request")

        if 'infer' in message:
            newPort = 0

            while True:
                newPort = getNewFreePort()

                if newPort != 0:
                    break

            th = threading.Thread(name='Port %d' % newPort,
                                  target=inference, args=(newPort, classifier))
            th.start()

            print('Sending new port back to ', message.split('from')[1])
            socket.send(("Request assigned port: " +
                         str(newPort)).encode('utf-8'))

        elif 'metrics' in message:
            print('Sending last metrics to ', message.split('from')[1])

            with open(metrics_dir+'_metrics.txt', 'r') as text_file:
                metrics = text_file.read()

            with open(metrics_dir+'_cm.txt', 'r') as text_file:
                cm = text_file.read()

            with open(metrics_dir+'_confusion_matrix.png', 'rb') as image_file:
                cm_pic = base64.b64encode(image_file.read())

            with open(metrics_dir+'_log_error.png', 'rb') as image_file:
                log_error_pic = base64.b64encode(image_file.read())

            ret = {
                'metrics': metrics,
                'cm': cm,
                'cm_pic': cm_pic.decode('utf-8'),
                'log_error_pic': log_error_pic.decode('utf-8')
            }

            socket.send(json.dumps(ret).encode('utf-8'))
