import tensorflow as tf
import dataHandler as dh
import sys

def _bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))

def _float_feature(value):
  return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

def _int_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

directory = 'dataset/tfrecords/'
paths_name = 'dataset/groups/*.json'

dataset_path = 'dataset/'
dictionary_path = dataset_path + 'dictionaries/'
dictionary_tags_path = dictionary_path + 'tags.txt'

validation_filename = directory+'validation.tfrecord'
train_filename = directory+'train.tfrecord'
test_filename = directory+'test.tfrecord'


def _writeRecord(filename, tupla, name):
    
    writer = tf.python_io.TFRecordWriter(filename)

    for index in range(len(tupla['labels'])):
        
        print(name,' data: {}/{}'.format(index+1, len(tupla['labels'])))

        feature = {
            'labels':  _int_feature(tupla['labels'][index]),
            'name': _bytes_feature(tupla['segments'][index]['name']),
            'tags': _bytes_feature(tupla['segments'][index]['tags']),
            'prevIns':        _bytes_feature(tupla['segments'][index]['previous']),
            'prevInsBool':    _int_feature(0 if b'None' in tupla['segments'][index]['previous'] else 1),
            'nxtIns':         _bytes_feature(tupla['segments'][index]['next']),
            'nxtInsBool':     _int_feature(0 if b'None' in tupla['segments'][index]['next'] else 1),
        }

        tag_list = dh.getDictionaryList(dictionary_tags_path)

        for tg in tag_list:
            feature[tg] = _int_feature(int(tupla['segments'][index][tg]))

        print(feature)
        
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

    _writeRecord(train_filename, train_tupla, "Train")
    _writeRecord(validation_filename, validation_tupla, "Validation")
    _writeRecord(test_filename, test_tupla, "Test")

    ds = tf.data.TFRecordDataset(test_filename)
    n = ds.make_one_shot_iterator().get_next()
    
    sess = tf.Session()
    print(sess.run(n))


