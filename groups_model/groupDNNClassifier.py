import dataHandler as dH
import math
import numpy as np
import tensorflow as tf
import pandas as pd
import os
import glob
import seaborn as sns
import itertools
from sklearn import metrics
from IPython import display
import matplotlib.pyplot as plt
from datetime import datetime

tf.logging.set_verbosity(tf.logging.ERROR)
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'

classes = 2
dataset_path = 'dataset/'

dictionary_path = dataset_path + 'dictionaries/'

dictionary_tags_path = dictionary_path + 'tags.txt'
dictionary_names_path = dictionary_path + 'names.txt'

records_path = dataset_path + 'tfrecords/'

train_tfrecord = records_path + 'train.tfrecord'
validation_tfrecord = records_path + 'validation.tfrecord'
test_tfrecord = records_path + 'test.tfrecord'

def _parse_function(record):
    
    tag_list = dH.getDictionaryList(dictionary_tags_path)

    features = {
        "labels": tf.FixedLenFeature([1], dtype=tf.int64),
        "name": tf.VarLenFeature(dtype=tf.string),
        "tags": tf.VarLenFeature(dtype=tf.string),
        "prevIns": tf.VarLenFeature(dtype=tf.string),
        "prevInsBool": tf.FixedLenFeature([1], dtype=tf.int64),
        "nxtIns": tf.VarLenFeature(dtype=tf.string),
        "nxtInsBool": tf.FixedLenFeature([1], dtype=tf.int64),
    }

    for tag in tag_list:
        features[tag] = tf.FixedLenFeature([1], dtype=tf.int64)

    parsed_features = tf.parse_single_example(record, features)

    labels = parsed_features['labels']

    name = parsed_features['name'].values
    tags = parsed_features['tags'].values

    link = parsed_features['link']
    meta = parsed_features['meta']
    style = parsed_features['style']
    title = parsed_features['title']
    body = parsed_features['body']
    address = parsed_features['address']
    article = parsed_features['article']
    aside = parsed_features['aside']
    footer = parsed_features['footer']
    header = parsed_features['header']
    h1 = parsed_features['h1']
    h2 = parsed_features['h2']
    h3 = parsed_features['h3']
    h4 = parsed_features['h4']
    h5 = parsed_features['h5']
    h6 = parsed_features['h6']
    hgroup = parsed_features['hgroup']
    nav = parsed_features['nav']
    section = parsed_features['section']
    blockquote = parsed_features['blockquote']
    dd = parsed_features['dd']
    dir_ = parsed_features['dir']
    div = parsed_features['div']
    dl = parsed_features['dl']
    dt = parsed_features['dt']
    figcaptio = parsed_features['figcaptio']
    figure = parsed_features['figure']
    hr = parsed_features['hr']
    li = parsed_features['li']
    main = parsed_features['main']
    ol = parsed_features['ol']
    p = parsed_features['p']
    pre = parsed_features['pre']
    ul = parsed_features['ul']
    a = parsed_features['a']
    abbr = parsed_features['abbr']
    b = parsed_features['b']
    bdi = parsed_features['bdi']
    bdo = parsed_features['bdo']
    br = parsed_features['br']
    cite = parsed_features['cite']
    code = parsed_features['code']
    data = parsed_features['data']
    dfn = parsed_features['dfn']
    em = parsed_features['em']
    i = parsed_features['i']
    kbd = parsed_features['kbd']
    mark = parsed_features['mark']
    nobr = parsed_features['nobr']
    q = parsed_features['q']
    rp = parsed_features['rp']
    rt = parsed_features['rt']
    rtc = parsed_features['rtc']
    ruby = parsed_features['ruby']
    s = parsed_features['s']
    samp = parsed_features['samp']
    small = parsed_features['small']
    span = parsed_features['span']
    strong = parsed_features['strong']
    sub = parsed_features['sub']
    time = parsed_features['time']
    tt = parsed_features['tt']
    u = parsed_features['u']
    var = parsed_features['var']
    wbr = parsed_features['wbr']
    area = parsed_features['area']
    audio = parsed_features['audio']
    img = parsed_features['img']
    map_ = parsed_features['map']
    track = parsed_features['track']
    video = parsed_features['video']
    applet = parsed_features['applet']
    embed = parsed_features['embed']
    noembed = parsed_features['noembed']
    object_ = parsed_features['object']
    picture = parsed_features['picture']
    param = parsed_features['param']
    source = parsed_features['source']
    canvas = parsed_features['canvas']
    noscript = parsed_features['noscript']
    script = parsed_features['script']
    del_ = parsed_features['del']
    ins = parsed_features['ins']
    caption = parsed_features['caption']
    col = parsed_features['col']
    colgroup = parsed_features['colgroup']
    table = parsed_features['table']
    tbody = parsed_features['tbody']
    td = parsed_features['td']
    tfoot = parsed_features['tfoot']
    th = parsed_features['th']
    thead = parsed_features['thead']
    tr = parsed_features['tr']
    button = parsed_features['button']
    datalist = parsed_features['datalist']
    fieldset = parsed_features['fieldset']
    form = parsed_features['form']
    input_ = parsed_features['input']
    label = parsed_features['label']
    legend = parsed_features['legend']
    meter = parsed_features['meter']
    optgroup = parsed_features['optgroup']
    option = parsed_features['option']
    output = parsed_features['output']
    progress = parsed_features['progress']
    select = parsed_features['select']
    textarea = parsed_features['textarea']
    details = parsed_features['details']
    dialog = parsed_features['dialog']
    menu = parsed_features['menu']
    menuitem = parsed_features['menuitem']
    summary = parsed_features['summary']
    path = parsed_features['path']
    g  = parsed_features['g']

    prevIns = parsed_features['prevIns'].values
    prevInsBool = parsed_features['prevInsBool']
    nxtIns = parsed_features['nxtIns'].values
    nxtInsBool = parsed_features['nxtInsBool']



    return {
        'name': name,
        'tags': tags,
        'prevIns': prevIns, 
        'prevInsBool': prevInsBool, 
        'nxtIns': nxtIns, 
        'nxtInsBool': nxtInsBool,        
        'link': link,
        'meta': meta,
        'style': style,
        'title': title,
        'body': body,
        'address': address,
        'article': article,
        'aside': aside,
        'footer': footer,
        'header': header,
        'h1': h1,
        'h2': h2,
        'h3': h3,
        'h4': h4,
        'h5': h5,
        'h6': h6,
        'hgroup': hgroup,
        'nav': nav,
        'section': section,
        'blockquote': blockquote,
        'dd': dd,
        'dir': dir_, 
        'div': div, 
        'dl': dl, 
        'dt': dt, 
        'figcaptio': figcaptio, 
        'figure': figure, 
        'hr': hr, 
        'li': li, 
        'main': main, 
        'ol': ol, 
        'p': p, 
        'pre': pre, 
        'ul': ul, 
        'a': a, 
        'abbr': abbr, 
        'b': b, 
        'bdi': bdi, 
        'bdo': bdo, 
        'br': br, 
        'cite': cite, 
        'code': code,
        'data': data, 
        'dfn': dfn, 
        'em': em, 
        'i': i, 
        'kbd': kbd, 
        'mark': mark, 
        'nobr': nobr, 
        'q': q, 
        'rp': rp, 
        'rt': rt, 
        'rtc': rtc,
        'ruby': ruby, 
        's': s, 
        'samp': samp, 
        'small': small, 
        'span': span, 
        'strong': strong, 
        'sub': sub, 
        'time': time, 
        'tt': tt, 
        'u': u, 
        'var': var, 
        'wbr': wbr, 
        'area': area, 
        'audio': audio, 
        'img': img, 
        'map': map_, 
        'track': track, 
        'video': video, 
        'applet': applet, 
        'embed': embed, 
        'noembed': noembed, 
        'object': object_, 
        'picture': picture, 
        'param': param, 
        'source': source,
        'canvas': canvas, 
        'noscript': noscript, 
        'script': script,
        'del': del_, 
        'ins': ins, 
        'caption': caption,
        'col': col, 
        'colgroup': colgroup, 
        'table': table, 
        'tbody': tbody, 
        'td': td, 
        'tfoot': tfoot, 
        'th': th, 
        'thead': thead, 
        'tr': tr, 
        'button': button, 
        'datalist': datalist, 
        'fieldset': fieldset, 
        'form': form, 
        'input': input_ , 
        'label': label, 
        'legend': legend, 
        'meter': meter, 
        'optgroup': optgroup, 
        'option': option, 
        'output': output, 
        'progress': progress, 
        'select': select, 
        'textarea': textarea, 
        'details': details, 
        'dialog': dialog, 
        'menu': menu, 
        'menuitem': menuitem, 
        'summary': summary, 
        'path': path, 
        'g': g   
    }, labels

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

def get_targets(input_filename, targets, batch_size, num_epoch=None):

    
        ds = tf.data.TFRecordDataset(input_filename)
        ds = ds.map(_parse_function)

        ds = ds.padded_batch(batch_size, ds.output_shapes)
        ds = ds.repeat(num_epoch)

        features, labels = ds.make_one_shot_iterator().get_next()

        with tf.Session() as sess:
            labs = sess.run(labels)
        
        targets.append(labs)


def process_targets(targets):

    ret = np.zeros(shape=(len(targets[0]), classes), dtype=float)

    for index, target in enumerate(targets[0]):        
        zeros = np.zeros((classes,))
        zeros[target] = 1
        ret[index] = zeros

    return ret       

def train_DNN_classification_model(
        learning_rate,
        steps,
        batch_size,
        periods,
        hidden_units,
        save_dir,
        time
    ):

    steps_per_period = steps/periods

    dictionary_names = dH.getDictionary(dictionary_names_path)
    dictionary_tags = dH.getDictionary(dictionary_tags_path)


    name_feature_column = tf.feature_column.categorical_column_with_vocabulary_list(
        key="name", 
        vocabulary_list=dictionary_names
    )
    name_embbeding_column = tf.feature_column.embedding_column(
        name_feature_column, 
        dimension = round(len(dictionary_names) ** 0.25)
    )

    tags_feature_column = tf.feature_column.categorical_column_with_vocabulary_list(
        key="tags",
        vocabulary_list=dictionary_tags
    )
    tags_embbeding_column = tf.feature_column.embedding_column(
        tags_feature_column, 
        dimension = round(len(dictionary_tags) ** 0.25)
    )

    prevIns_feature_column = tf.feature_column.categorical_column_with_vocabulary_list(
        key="prevIns", 
        vocabulary_list=dictionary_names
    )
    prevIns_embbeding_column = tf.feature_column.embedding_column(
        prevIns_feature_column, 
        dimension = round(len(dictionary_names) ** 0.25)
    )

    nxtIns_feature_column = tf.feature_column.categorical_column_with_vocabulary_list(
        key="nxtIns", 
        vocabulary_list=dictionary_names
    )
    nxtIns_embbeding_column = tf.feature_column.embedding_column(
        nxtIns_feature_column, 
        dimension = round(len(dictionary_names) ** 0.25)
    )
    
    my_optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    my_optimizer = tf.contrib.estimator.clip_gradients_by_norm(my_optimizer, 5.0)

    feature_columns = [
        name_embbeding_column,
        tags_embbeding_column,
        prevIns_embbeding_column,
        tf.feature_column.numeric_column('prevInsBool', shape=1),
        nxtIns_embbeding_column, 
        tf.feature_column.numeric_column('nxtInsBool', shape=1)
    ]

    tag_list = dH.getDictionary(dictionary_tags_path)

    for tg in tag_list:
        feature_columns.append(tf.feature_column.numeric_column(tg, shape=1))
    
    DNN_class = tf.estimator.DNNClassifier(
        feature_columns = feature_columns,
        n_classes=classes,
        hidden_units= hidden_units,
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

    print("\n=> Training the model...\n\nLogLoss error (on validation data):")
    training_errors = []
    validation_errors = []

    for period in range(0, periods):

        DNN_class.train(
            input_fn = training_input_fn,
            steps=steps_per_period
        )

        # Take a break and compute probabilities.
        training_predictions = list(DNN_class.predict(input_fn=training_predict_fn))
        # training_probabilities = np.array([item['probabilities'] for item in training_predictions])
        training_pred_class_id = np.array([item['class_ids'][0] for item in training_predictions])
        training_pred_one_hot = tf.keras.utils.to_categorical(training_pred_class_id, num_classes=classes)

        training_targets = []
        get_targets([train_tfrecord], training_targets, len(training_pred_one_hot), 1)
        training_targets = process_targets(training_targets)

        validation_predictions = list(DNN_class.predict(input_fn=validation_predict_fn))
        # validation_probabilities = np.array([item['probabilities'] for item in validation_predictions])    
        validation_pred_class_id = np.array([item['class_ids'][0] for item in validation_predictions])
        validation_pred_one_hot = tf.keras.utils.to_categorical(validation_pred_class_id, num_classes=classes)    
        
        validation_targets = []
        get_targets([validation_tfrecord], validation_targets, len(validation_pred_one_hot), 1)
        validation_targets = process_targets(validation_targets)

        # Compute training and validation errors.
        training_log_loss = metrics.log_loss(training_targets, training_pred_one_hot)
        validation_log_loss = metrics.log_loss(validation_targets, validation_pred_one_hot)
        # Occasionally print the current loss.
        print("\n  - Period %02d/%02d : %0.2f" % (period+1, periods , validation_log_loss))
        # Add the loss metrics from this period to our list.
        training_errors.append(training_log_loss)
        validation_errors.append(validation_log_loss)
    
    print("\n=> Model training finished.\n")
    # Remove event files to save disk space.
    _ = map(os.remove, glob.glob(os.path.join(DNN_class.model_dir, 'events.out.tfevents*')))
    
    validation_targets = []
    get_targets([validation_tfrecord], validation_targets, len(validation_pred_one_hot), 1)
    validation_targets = process_targets(validation_targets)

    # Calculate final predictions (not probabilities, as above).
    final_predictions = DNN_class.predict(input_fn=validation_predict_fn)
    final_predictions = np.array([item['class_ids'][0] for item in final_predictions])
    final_predictions_class_id = np.array([item['class_ids'][0] for item in validation_predictions])
    final_predictions_one_hot = tf.keras.utils.to_categorical(final_predictions_class_id, num_classes=classes)  

    print("Final pred",final_predictions_one_hot, "\nVal targs: ", validation_targets)

    accuracy = metrics.accuracy_score(validation_targets, final_predictions_one_hot)
    print("Final accuracy (on validation data): %0.2f" % accuracy)  

    # Output a graph of loss metrics over periods.
    plt.ylabel("LogLoss")
    plt.xlabel("Periods")
    plt.title("LogLoss vs. Periods")
    plt.plot(training_errors, label="training")
    plt.plot(validation_errors, label="validation") 
    plt.legend()
    plt.savefig(save_dir+time+"_log_error.png")
    plt.show()
    
    #y_test.values.argmax(axis=1), predictions.argmax(axis=1)

    # Output a plot of the confusion matrix.
    cm = metrics.confusion_matrix(validation_targets.argmax(axis=1), final_predictions)
    # Normalize the confusion matrix by row (i.e by the number of samples
    # in each class)
    cm_normalized = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
    ax = sns.heatmap(cm_normalized, cmap="bone_r")
    ax.set_aspect(1)
    plt.title("Confusion matrix")
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.savefig(save_dir+time+"_confusion_matrix.png")
    plt.show()

    return DNN_class

def run(learning_rate, steps, batch_size, periods, hidden_units, save_dir):
    
    time = dH._replaceMultiple(str(datetime.now()).split('.')[0], [' ',':','-'], '_')

    #Change values
    classifier = train_DNN_classification_model(
                learning_rate=learning_rate,
                steps=steps,
                batch_size=batch_size,
                periods=periods,
                hidden_units=hidden_units,
                save_dir=save_dir,
                time=time)

    settings = {
        'learning_rate': learning_rate,
        'steps': steps,
        'batch_size': batch_size,
        'periods': periods,
        'hidden_units': hidden_units
    }

    with open(save_dir+time+"_settings.txt", "w") as text_file:
            print(f"{settings}", file=text_file)

    print("\n=> Evaluating model...\n")

    metrics = ''

    #Evaluate over the training set
    evaluation_metrics = classifier.evaluate(
        input_fn= create_input_fn([train_tfrecord],
                                    batch_size,
                                num_epoch=1,
                                shuffle=True))

    print ("Training set metrics:")
    for m in evaluation_metrics:
        print (m, evaluation_metrics[m])
        metrics += m + ' ' + str(evaluation_metrics[m]) + '\n'
    print ("\n---\n")

    #Evaluate over the validation set
    evaluation_metrics = classifier.evaluate(
        input_fn= create_input_fn([validation_tfrecord],
                                    batch_size,
                                num_epoch=1,
                                shuffle=True))

    print ("Validation set metrics:")
    for m in evaluation_metrics:
        print (m, evaluation_metrics[m])
        metrics += m + ' ' + str(evaluation_metrics[m]) + '\n'
    print ("\n---\n")

    #Evaluate over the test set
    evaluation_metrics = classifier.evaluate(
        input_fn= create_input_fn([test_tfrecord],
                                    batch_size,
                                num_epoch=1,
                                shuffle=True))

    print ("Test set metrics:")
    for m in evaluation_metrics:
        print (m, evaluation_metrics[m])
        metrics += m + ' ' + str(evaluation_metrics[m]) + '\n'
    print ("\n---\n")

    with open(save_dir+time+"_metrics.txt", "w") as text_file:
            print(f"{metrics}", file=text_file)