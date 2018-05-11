import glob
import itertools
import json
import math
import os
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from IPython import display
from sklearn import metrics

from operator import itemgetter

import dataHandler as dH

# Define verbosity
tf.logging.set_verbosity(tf.logging.ERROR)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Define num of classes to estimate
classes = 15

# Define dataset, dictionaries
dataset_path = 'dataset/'
dictionary_path = dataset_path + 'dictionaries/'
dictionary_classes_path = dictionary_path + 'classes.txt'
dictionary_text_path = dictionary_path + 'text.txt'
dictionary_symbols_path = dictionary_path + 'symbols.txt'
dictionary_colors_path = dictionary_path + 'colors.txt'
records_path = dataset_path + 'tfrecords/'
train_tfrecord = records_path + 'train.tfrecord'
validation_tfrecord = records_path + 'validation.tfrecord'
test_tfrecord = records_path + 'test.tfrecord'


'''
Input: Tfrecord unparsed
Returns: Dict with features and labels parsed
- Parses a tfrecord to feed the model
'''


def _parse_function(record):

    features = {
        'labels':       tf.FixedLenFeature([1], dtype=tf.int64),
        'insName':      tf.VarLenFeature(dtype=tf.string),
        'insSymID':     tf.VarLenFeature(dtype=tf.string),
        'insSymIDBool': tf.FixedLenFeature([1], dtype=tf.int64),
        'insColor':     tf.VarLenFeature(dtype=tf.string),
        'insColorBool': tf.FixedLenFeature([1], dtype=tf.int64),
        'insText':      tf.VarLenFeature(dtype=tf.string),
        'insFrameClass':  tf.VarLenFeature(dtype=tf.string),
        'insFrameHeight': tf.FixedLenFeature([1], dtype=tf.float32),
        'insFrameWidth': tf.FixedLenFeature([1], dtype=tf.float32),
        'prevIns': tf.VarLenFeature(dtype=tf.string),
        'prevInsBool': tf.FixedLenFeature([1], dtype=tf.int64),
        'nxtIns': tf.VarLenFeature(dtype=tf.string),
        'nxtInsBool': tf.FixedLenFeature([1], dtype=tf.int64),
        'parent': tf.VarLenFeature(dtype=tf.string),
        'parentBool': tf.FixedLenFeature([1], dtype=tf.int64),
    }

    for i in range(1, 6):
        obj = 'obj' + str(i)
        features[obj+'Name'] = tf.VarLenFeature(dtype=tf.string)
        features[obj+'Class'] = tf.VarLenFeature(dtype=tf.string)
        features[obj+'Bool'] = tf.FixedLenFeature([1], dtype=tf.int64)

    parsed_features = tf.parse_single_example(record, features)

    labels = parsed_features['labels']

    insName = parsed_features['insName'].values
    insSymID = parsed_features['insSymID'].values
    insColor = parsed_features['insColor'].values
    insText = parsed_features['insText'].values
    insFrameClass = parsed_features['insFrameClass'].values
    prevIns = parsed_features['prevIns'].values
    nxtIns = parsed_features['nxtIns'].values
    parent = parsed_features['parent'].values
    obj1Name = parsed_features['obj1Name'].values
    obj2Name = parsed_features['obj2Name'].values
    obj3Name = parsed_features['obj3Name'].values
    obj4Name = parsed_features['obj4Name'].values
    obj5Name = parsed_features['obj5Name'].values
    obj1Class = parsed_features['obj1Class'].values
    obj2Class = parsed_features['obj2Class'].values
    obj3Class = parsed_features['obj3Class'].values
    obj4Class = parsed_features['obj4Class'].values
    obj5Class = parsed_features['obj5Class'].values

    insSymIDBool = parsed_features['insSymIDBool']
    nxtInsBool = parsed_features['nxtInsBool']
    prevInsBool = parsed_features['prevInsBool']
    insColorBool = parsed_features['insColorBool']
    parentBool = parsed_features['parentBool']
    obj1Bool = parsed_features['obj1Bool']
    obj2Bool = parsed_features['obj2Bool']
    obj3Bool = parsed_features['obj3Bool']
    obj4Bool = parsed_features['obj4Bool']
    obj5Bool = parsed_features['obj5Bool']

    insFrameHeight = parsed_features['insFrameHeight']
    insFrameWidth = parsed_features['insFrameWidth']

    return {
        'insName': insName,
        'insSymID': insSymID,
        'insSymIDBool': insSymIDBool,
        'insColor': insColor,
        'insColorBool': insColorBool,
        'insText': insText,
        'insFrameClass': insFrameClass,
        'insFrameHeight': insFrameHeight,
        'insFrameWidth': insFrameWidth,
        'prevIns': prevIns,
        'prevInsBool': prevInsBool,
        'nxtIns': nxtIns,
        'nxtInsBool': nxtInsBool,
        'parent': parent,
        'parentBool': parentBool,
        'obj1Name': obj1Name,
        'obj1Class': obj1Class,
        'obj1Bool': obj1Bool,
        'obj2Name': obj2Name,
        'obj2Class': obj2Class,
        'obj2Bool': obj2Bool,
        'obj3Name': obj3Name,
        'obj3Class': obj3Class,
        'obj3Bool': obj3Bool,
        'obj4Name': obj4Name,
        'obj4Class': obj4Class,
        'obj4Bool': obj4Bool,
        'obj5Name': obj5Name,
        'obj5Class': obj5Class,
        'obj5Bool': obj5Bool
    }, labels


'''
Input: Tfrecord path (str), batch size (int), num epoch (int optional, defaults to None), shuffle (bool optional, defaults to True)
Returns: Function to feed the train and evaluate functions of the model
- Load & parses a tfrecord, shuffles it if defined, pad it, repeats it if defined and return features & labels as a function
'''


def create_input_fn(input_filename, batch_size, num_epoch=None, shuffle=True):

    def _input_fn():

        ds = tf.data.TFRecordDataset(input_filename)
        ds = ds.map(_parse_function)

        if shuffle:
            ds = ds.shuffle(10000)

        ds = ds.padded_batch(batch_size, ds.output_shapes)
        ds = ds.repeat(num_epoch)

        features, labels = ds.make_one_shot_iterator().get_next()

        return features, labels

    return _input_fn


'''
Input: Tfrecord path (str), batch size (int), num epoch (int optional, defaults to 1)
Returns: Function to feed the predict function of the model
- Load & parses a tfrecord, pad it, repeats it if defined and return features as a function
'''


def create_input_feat_fn(input_filename, batch_size, num_epoch=1):

    def _input_fn():

        ds = tf.data.TFRecordDataset(input_filename)
        ds = ds.map(_parse_function)

        ds = ds.padded_batch(batch_size, ds.output_shapes)
        ds = ds.repeat(num_epoch)

        features, _ = ds.make_one_shot_iterator().get_next()

        return features

    return _input_fn


'''
Input: Tfrecord path (str), batch size (int), num epoch (int optional, defaults to 1)
Returns: Features tensor
- Load & parses a tfrecord, pad it, repeats it if defined and return features as a tensor
'''


def get_features(input_filename, batch_size, num_epoch=1):

    ds = tf.data.TFRecordDataset(input_filename)
    ds = ds.map(_parse_function)

    ds = ds.padded_batch(batch_size, ds.output_shapes)
    ds = ds.repeat(num_epoch)

    features, _ = ds.make_one_shot_iterator().get_next()

    with tf.Session() as sess:
        feats = sess.run(features)

    return feats


'''
Input: Tfrecord path (str), batch size (int), num epoch (int optional, defaults to 1), targets (empty list to append the targets)
- Load & parses a tfrecord, pad it and appends to targets the labels to compute the error
'''


def get_targets(input_filename, targets, batch_size, num_epoch=1):

    ds = tf.data.TFRecordDataset(input_filename)
    ds = ds.map(_parse_function)

    ds = ds.padded_batch(batch_size, ds.output_shapes)
    ds = ds.repeat(num_epoch)

    _, labels = ds.make_one_shot_iterator().get_next()

    with tf.Session() as sess:
        labs = sess.run(labels)

    targets.append(labs)


'''
Inputs: targets list
Returns: targets numpy array in one-hot
'''


def process_targets(targets):

    ret = np.zeros(shape=(len(targets[0]), classes), dtype=float)

    for index, target in enumerate(targets[0]):
        zeros = np.zeros((classes,))
        zeros[target] = 1
        ret[index] = zeros

    return ret


'''
Inputs: Features values list, Number of buckets (int)
Returns: Quantiles list with len equal to number of buckets
'''


def get_quantile_based_buckets(feature_values, num_buckets):

    quantiles = feature_values.quantile(
        [(i+1.)/(num_buckets + 1.) for i in range(num_buckets)])

    return [quantiles[q] for q in quantiles.keys()]


'''
Inputs: learning_rate (float), steps (int), batch_size (int), periods (int), hidden_units (list), metrics_dir (str)
Returns: a trained dnn classifier according to the parameters
- Load dictionaries, get quantiles for sizes, bucketized size according to quantiles, map strings in one-hot with vocabulary list,
  embbeds one-hot strings, defines the DNN classifier, build the input functions to train and evaluate, train and evaluate in periods,
  save confusion matrix pic and data, saves log error pic and returns the classifier
'''


def train_DNN_classification_model(
    learning_rate,
    steps,
    batch_size,
    periods,
    hidden_units,
    metrics_dir
):

    steps_per_period = steps/periods
    dictionary_classes = dH.getDictionary(dictionary_classes_path)
    dictionary_colors = dH.getDictionary(dictionary_colors_path)
    dictionary_text = dH.getDictionary(dictionary_text_path)
    dictionary_symbols = dH.getDictionary(dictionary_symbols_path)

    feat = get_features([train_tfrecord], 10000)

    widthPandas = pd.DataFrame(feat['insFrameWidth'])
    heightPandas = pd.DataFrame(feat['insFrameHeight'])

    bound_width = get_quantile_based_buckets(widthPandas, 25)[0].tolist()
    bound_height = get_quantile_based_buckets(heightPandas, 25)[0].tolist()

    bound_width = sorted(list(set(bound_width)))
    bound_height = sorted(list(set(bound_height)))

    height_numeric_column = tf.feature_column.numeric_column(
        'insFrameHeight', shape=1)
    height_bucketized_column = tf.feature_column.bucketized_column(
        source_column=height_numeric_column,
        boundaries=bound_height
    )

    width_numeric_column = tf.feature_column.numeric_column(
        'insFrameWidth', shape=1)
    width_bucketized_column = tf.feature_column.bucketized_column(
        source_column=width_numeric_column,
        boundaries=bound_width
    )

    insName_feature_column = tf.feature_column.categorical_column_with_vocabulary_list(
        key='insName',
        vocabulary_list=dictionary_text
    )
    insName_embbeding_column = tf.feature_column.embedding_column(
        insName_feature_column,
        dimension=round(len(dictionary_text) ** 0.25)
    )

    insSymID_feature_column = tf.feature_column.categorical_column_with_vocabulary_list(
        key='insSymID',
        vocabulary_list=dictionary_symbols
    )
    insSymID_embbeding_column = tf.feature_column.embedding_column(
        insSymID_feature_column,
        dimension=round(len(dictionary_symbols) ** 0.25)
    )

    insColor_feature_column = tf.feature_column.categorical_column_with_vocabulary_list(
        key='insColor',
        vocabulary_list=dictionary_colors
    )
    insColor_embbeding_column = tf.feature_column.embedding_column(
        insColor_feature_column,
        dimension=round(len(dictionary_colors) ** 0.25)
    )

    insText_feature_column = tf.feature_column.categorical_column_with_vocabulary_list(
        key='insText',
        vocabulary_list=dictionary_text
    )
    insText_embbeding_column = tf.feature_column.embedding_column(
        insText_feature_column,
        dimension=round(len(dictionary_text) ** 0.25)
    )

    insFrameClass_feature_column = tf.feature_column.categorical_column_with_vocabulary_list(
        key='insFrameClass',
        vocabulary_list=dictionary_classes
    )
    insFrameClass_embbeding_column = tf.feature_column.embedding_column(
        insFrameClass_feature_column,
        dimension=round(len(dictionary_classes) ** 0.25)
    )

    prevIns_feature_column = tf.feature_column.categorical_column_with_vocabulary_list(
        key='prevIns',
        vocabulary_list=dictionary_text
    )
    prevIns_embbeding_column = tf.feature_column.embedding_column(
        prevIns_feature_column,
        dimension=round(len(dictionary_text) ** 0.25)
    )

    nxtIns_feature_column = tf.feature_column.categorical_column_with_vocabulary_list(
        key='nxtIns',
        vocabulary_list=dictionary_text
    )
    nxtIns_embbeding_column = tf.feature_column.embedding_column(
        nxtIns_feature_column,
        dimension=round(len(dictionary_text) ** 0.25)
    )

    parent_feature_column = tf.feature_column.categorical_column_with_vocabulary_list(
        key='parent',
        vocabulary_list=dictionary_text
    )
    parent_embbeding_column = tf.feature_column.embedding_column(
        parent_feature_column,
        dimension=round(len(dictionary_text) ** 0.25)
    )

    # 1
    obj1Name_feature_column = tf.feature_column.categorical_column_with_vocabulary_list(
        key='obj1Name',
        vocabulary_list=dictionary_text
    )
    obj1Name_embbeding_column = tf.feature_column.embedding_column(
        obj1Name_feature_column,
        dimension=round(len(dictionary_text) ** 0.25)
    )
    obj1Class_feature_column = tf.feature_column.categorical_column_with_vocabulary_list(
        key='obj1Class',
        vocabulary_list=dictionary_classes
    )
    obj1Class_embbeding_column = tf.feature_column.embedding_column(
        obj1Class_feature_column,
        dimension=round(len(dictionary_classes) ** 0.25)
    )

    # 2
    obj2Name_feature_column = tf.feature_column.categorical_column_with_vocabulary_list(
        key='obj2Name',
        vocabulary_list=dictionary_text
    )
    obj2Name_embbeding_column = tf.feature_column.embedding_column(
        obj2Name_feature_column,
        dimension=round(len(dictionary_text) ** 0.25)
    )
    obj2Class_feature_column = tf.feature_column.categorical_column_with_vocabulary_list(
        key='obj2Class',
        vocabulary_list=dictionary_classes
    )
    obj2Class_embbeding_column = tf.feature_column.embedding_column(
        obj2Class_feature_column,
        dimension=round(len(dictionary_classes) ** 0.25)
    )

    # 3
    obj3Name_feature_column = tf.feature_column.categorical_column_with_vocabulary_list(
        key='obj3Name',
        vocabulary_list=dictionary_text
    )
    obj3Name_embbeding_column = tf.feature_column.embedding_column(
        obj3Name_feature_column,
        dimension=round(len(dictionary_text) ** 0.25)
    )
    obj3Class_feature_column = tf.feature_column.categorical_column_with_vocabulary_list(
        key='obj3Class',
        vocabulary_list=dictionary_classes
    )
    obj3Class_embbeding_column = tf.feature_column.embedding_column(
        obj3Class_feature_column,
        dimension=round(len(dictionary_classes) ** 0.25)
    )

    # 4
    obj4Name_feature_column = tf.feature_column.categorical_column_with_vocabulary_list(
        key='obj4Name',
        vocabulary_list=dictionary_text
    )
    obj4Name_embbeding_column = tf.feature_column.embedding_column(
        obj4Name_feature_column,
        dimension=round(len(dictionary_text) ** 0.25)
    )
    obj4Class_feature_column = tf.feature_column.categorical_column_with_vocabulary_list(
        key='obj4Class',
        vocabulary_list=dictionary_classes
    )
    obj4Class_embbeding_column = tf.feature_column.embedding_column(
        obj4Class_feature_column,
        dimension=round(len(dictionary_classes) ** 0.25)
    )

    # 5
    obj5Name_feature_column = tf.feature_column.categorical_column_with_vocabulary_list(
        key='obj5Name',
        vocabulary_list=dictionary_text
    )
    obj5Name_embbeding_column = tf.feature_column.embedding_column(
        obj5Name_feature_column,
        dimension=round(len(dictionary_text) ** 0.25)
    )
    obj5Class_feature_column = tf.feature_column.categorical_column_with_vocabulary_list(
        key='obj5Class',
        vocabulary_list=dictionary_classes
    )
    obj5Class_embbeding_column = tf.feature_column.embedding_column(
        obj5Class_feature_column,
        dimension=round(len(dictionary_classes) ** 0.25)
    )

    my_optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=0.9, use_nesterov=True)
    my_optimizer = tf.contrib.estimator.clip_gradients_by_norm(
        my_optimizer, 3.0)

    feature_columns = [
        insName_embbeding_column,
        insSymID_embbeding_column,
        tf.feature_column.numeric_column('insSymIDBool', shape=1),
        insColor_embbeding_column,
        tf.feature_column.numeric_column('insColorBool', shape=1),
        insText_embbeding_column,
        insFrameClass_embbeding_column,
        height_bucketized_column,
        width_bucketized_column,
        prevIns_embbeding_column,
        tf.feature_column.numeric_column('prevInsBool', shape=1),
        nxtIns_embbeding_column,
        tf.feature_column.numeric_column('nxtInsBool', shape=1),
        parent_embbeding_column,
        tf.feature_column.numeric_column('parentBool', shape=1),
        obj1Name_embbeding_column,
        obj1Class_embbeding_column,
        tf.feature_column.numeric_column('obj1Bool', shape=1),
        obj2Name_embbeding_column,
        obj2Class_embbeding_column,
        tf.feature_column.numeric_column('obj2Bool', shape=1),
        obj3Name_embbeding_column,
        obj3Class_embbeding_column,
        tf.feature_column.numeric_column('obj3Bool', shape=1),
        obj4Name_embbeding_column,
        obj4Class_embbeding_column,
        tf.feature_column.numeric_column('obj4Bool', shape=1),
        obj5Name_embbeding_column,
        obj5Class_embbeding_column,
        tf.feature_column.numeric_column('obj5Bool', shape=1)
    ]

    DNN_class = tf.estimator.DNNClassifier(
        feature_columns=feature_columns,
        n_classes=classes,
        hidden_units=hidden_units,
        optimizer=my_optimizer,
        config=tf.estimator.RunConfig(keep_checkpoint_max=1)
    )

    training_input_fn = create_input_fn([train_tfrecord],
                                        batch_size)

    training_predict_fn = create_input_fn([train_tfrecord],
                                          batch_size,
                                          num_epoch=1,
                                          shuffle=False)

    validation_predict_fn = create_input_fn([validation_tfrecord],
                                            batch_size,
                                            num_epoch=1,
                                            shuffle=False)

    print('\n=> Training the model...\n\nLogLoss error (on validation data):')
    training_errors = []
    validation_errors = []

    for period in range(0, periods):

        DNN_class.train(
            input_fn=training_input_fn,
            steps=steps_per_period
        )

        # Take a break and compute probabilities.
        training_predictions = list(
            DNN_class.predict(input_fn=training_predict_fn))
        # training_probabilities = np.array([item['probabilities'] for item in training_predictions])
        training_pred_class_id = np.array(
            [item['class_ids'][0] for item in training_predictions])
        training_pred_one_hot = tf.keras.utils.to_categorical(
            training_pred_class_id, num_classes=classes)

        training_targets = []
        get_targets([train_tfrecord], training_targets,
                    len(training_pred_one_hot))
        training_targets = process_targets(training_targets)

        validation_predictions = list(
            DNN_class.predict(input_fn=validation_predict_fn))
        # validation_probabilities = np.array([item['probabilities'] for item in validation_predictions])
        validation_pred_class_id = np.array(
            [item['class_ids'][0] for item in validation_predictions])
        validation_pred_one_hot = tf.keras.utils.to_categorical(
            validation_pred_class_id, num_classes=classes)

        validation_targets = []
        get_targets([validation_tfrecord], validation_targets,
                    len(validation_pred_one_hot))
        validation_targets = process_targets(validation_targets)

        # Compute training and validation errors.
        training_log_loss = metrics.log_loss(
            training_targets, training_pred_one_hot)
        validation_log_loss = metrics.log_loss(
            validation_targets, validation_pred_one_hot)
        # Occasionally print the current loss.
        print('\n  - Period %02d/%02d : %0.2f' %
              (period+1, periods, validation_log_loss))
        # Add the loss metrics from this period to our list.
        training_errors.append(training_log_loss)
        validation_errors.append(validation_log_loss)

    print('\n=> Model training finished.\n')
    # Remove event files to save disk space.
    _ = map(os.remove, glob.glob(os.path.join(
        DNN_class.model_dir, 'events.out.tfevents*')))

    validation_targets = []
    get_targets([validation_tfrecord], validation_targets,
                len(validation_pred_one_hot))
    validation_targets = process_targets(validation_targets)

    # Calculate final predictions (not probabilities, as above).
    final_predictions = DNN_class.predict(input_fn=validation_predict_fn)
    final_predictions = np.array([item['class_ids'][0]
                                  for item in final_predictions])
    final_predictions_class_id = np.array(
        [item['class_ids'][0] for item in validation_predictions])
    final_predictions_one_hot = tf.keras.utils.to_categorical(
        final_predictions_class_id, num_classes=classes)

    accuracy = metrics.accuracy_score(
        validation_targets, final_predictions_one_hot)
    print('Final accuracy (on validation data): %0.2f' % accuracy)

    # Output a graph of loss metrics over periods.
    plt.ylabel('LogLoss')
    plt.xlabel('Periods')
    plt.title('LogLoss vs. Periods')
    plt.plot(training_errors, label='training')
    plt.plot(validation_errors, label='validation')
    plt.legend()
    plt.savefig(metrics_dir+'_log_error.png')
    plt.close()

    # Output a plot of the confusion matrix.
    cm = metrics.confusion_matrix(
        validation_targets.argmax(axis=1), final_predictions)
    # Normalize the confusion matrix by row (i.e by the number of samples
    # in each class)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    cm_total, cm_correct, cm_incorr = dH.checkConfusionMatrix(cm_normalized)
    cm_data = 'Predicted number of classes: '+cm_total
    cm_data += '\nPredicted correctly: '+cm_correct + \
        '\nPredicted incorrectly: '+cm_incorr
    cm_data += '\nNot confused %: ' + \
        str(100*float(cm_correct)/float(cm_total))+'%'

    with open(metrics_dir+'_cm.txt', 'w') as text_file:
        print(f'{cm_data}', file=text_file)

    ax = sns.heatmap(cm_normalized, annot=True, cmap='Blues')
    ax.set_aspect(1)
    plt.title('Confusion matrix')
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig(metrics_dir+'_confusion_matrix.png')
    plt.close()

    return DNN_class


'''
Inputs: learning_rate (float), steps (int), batch_size (int), periods (int), hidden_units (list), save_dir (str)
Returns: a trained dnn classifier according to the parameters and the metrics directory
- Train and evaluate a new DNNClassifier, save the settings, reevaluate the model and save the results, returns the classifier and the save dir
'''


def train_and_evaluate(learning_rate, steps, batch_size, periods, hidden_units, save_dir):

    time = dH._replaceMultiple(str(datetime.now()).split('.')[
                               0], [' ', ':', '-'], '_')
    metrics_dir = save_dir+time

    classifier = train_DNN_classification_model(
        learning_rate=learning_rate,
        steps=steps,
        batch_size=batch_size,
        periods=periods,
        hidden_units=hidden_units,
        metrics_dir=metrics_dir)

    settings = {
        'learning_rate': learning_rate,
        'steps': steps,
        'batch_size': batch_size,
        'periods': periods,
        'hidden_units': hidden_units,
        'save_dir': save_dir,
        'time': time
    }

    settings = json.dumps(settings, indent=4, sort_keys=True)

    with open(metrics_dir+'_settings.txt', 'w') as text_file:
        print(f'{settings}', file=text_file)

    print('\n=> Evaluating model...\n')

    metrics = 'Training metrics:\n'

    # Evaluate over the training set
    evaluation_metrics = classifier.evaluate(
        input_fn=create_input_fn([train_tfrecord],
                                 batch_size,
                                 num_epoch=1,
                                 shuffle=True))

    print('Training set metrics:')
    for m in evaluation_metrics:
        print(m, evaluation_metrics[m])
        metrics += m + ' ' + str(evaluation_metrics[m]) + '\n'
    print('\n---\n')

    metrics += '\nValidation metrics:\n'

    # Evaluate over the validation set
    evaluation_metrics = classifier.evaluate(
        input_fn=create_input_fn([validation_tfrecord],
                                 batch_size,
                                 num_epoch=1,
                                 shuffle=True))

    print('Validation set metrics:')
    for m in evaluation_metrics:
        print(m, evaluation_metrics[m])
        metrics += m + ' ' + str(evaluation_metrics[m]) + '\n'
    print('\n---\n')

    metrics += '\nTest metrics:\n'

    # Evaluate over the test set
    evaluation_metrics = classifier.evaluate(
        input_fn=create_input_fn([test_tfrecord],
                                 batch_size,
                                 num_epoch=1,
                                 shuffle=True))

    print('Test set metrics:')
    for m in evaluation_metrics:
        print(m, evaluation_metrics[m])
        metrics += m + ' ' + str(evaluation_metrics[m]) + '\n'
    print('\n---\n')

    with open(metrics_dir+'_metrics.txt', 'w') as text_file:
        print(f'{metrics}', file=text_file)

    return classifier, metrics_dir
