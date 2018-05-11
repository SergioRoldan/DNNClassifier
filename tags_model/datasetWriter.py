import tensorflow as tf
import dataHandler as dh
import sys
import time

directory = 'dataset/tfrecords/'
paths_name = 'dataset/symbols/*.json'

validation_filename = directory+'validation.tfrecord'
train_filename = directory+'train.tfrecord'
test_filename = directory+'test.tfrecord'

def _bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))

def _float_feature(value):
  return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

def _int_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _writeRecord(filename, tupla, name):
    
    writer = tf.python_io.TFRecordWriter(filename)

    for index in range(len(tupla['labels'])):
        
        print(name,' data: {}/{}'.format(index+1, len(tupla['labels'])))

        feature = {
            'labels':         _int_feature(tupla['labels'][index]),
            'insName':        _bytes_feature(tupla['segments'][index]['name']),
            'insSymID':       _bytes_feature(tupla['segments'][index]['symbolID']),
            'insSymIDBool':   _int_feature(0 if b'none' in tupla['segments'][index]['symbolID'] else 1),
            'insColor':       _bytes_feature(tupla['segments'][index]['color']),
            'insColorBool':   _int_feature(0 if b'none' in tupla['segments'][index]['color'] else 1),
            'insText':        _bytes_feature(tupla['segments'][index]['text']),
            'insFrameClass':  _bytes_feature(tupla['segments'][index]['frame']['class']),
            'insFrameHeight': _float_feature(float(tupla['segments'][index]['frame']['height'])),
            'insFrameWidth':  _float_feature(float(tupla['segments'][index]['frame']['width'])),
            'prevIns':        _bytes_feature(tupla['segments'][index]['previous']),
            'prevInsBool':    _int_feature(0 if b'none' in tupla['segments'][index]['previous'] else 1),
            'nxtIns':         _bytes_feature(tupla['segments'][index]['next']),
            'nxtInsBool':     _int_feature(0 if b'none' in tupla['segments'][index]['next'] else 1),
            'parent':         _bytes_feature(tupla['segments'][index]['parent']),
            'parentBool':     _int_feature(0 if b'none' in tupla['segments'][index]['parent'] else 1),
        }

        for i in range (1,6):
            obj = 'obj' + str(i)
            feature[obj+'Name'] = _bytes_feature(tupla['segments'][index][obj]['name'])
            feature[obj+'Class'] = _bytes_feature(tupla['segments'][index][obj]['class'])
            feature[obj+'Bool'] = _int_feature(0 if b'none' in tupla['segments'][index][obj]['name'] else 1)

        '''if index == 0:
            print(feature)
            time.sleep(10)'''

        example = tf.train.Example(features=tf.train.Features(feature=feature))
        writer.write(example.SerializeToString())

    writer.close()
    sys.stdout.flush()

def run(train_percent, val_percent, percent):
    
    train_tupla, validation_tupla, test_tupla = dh.handleData(paths_name,
                                                            train_percent,
                                                            val_percent,
                                                            percent
                                                            )

    _writeRecord(train_filename, train_tupla, 'Train')
    _writeRecord(validation_filename, validation_tupla, 'Validation')
    _writeRecord(test_filename, test_tupla, 'Test')

    ds = tf.data.TFRecordDataset(train_filename)
    n = ds.make_one_shot_iterator().get_next()
    
    sess = tf.Session()
    print(sess.run(n))
