import re
import glob
import random
from pathlib import Path
import os
import json
import time
import itertools

dictionary_pattern = r'(?:\')+([\w#-]+)+(?:\')'
root_name = 'dataset/symbols'
consumed_root_name = 'consumed/symbols'
dicitonary_addr = 'dataset/dictionaries/'
labls = ['button', 'input', 'textarea', 'alert', 'table', 'footer', 'link', 'sidebar', 'status', 'paragraph', 'br', 'timeline', 'item', 'header']
minRep = 2

#Addresses from path
def _addrsFromPath(paths_name):
    return glob.glob(paths_name)

#Shuffle labels and segments    
def _shuffleData(labels, segments):
    
    c = list(zip(segments, labels))
    random.shuffle(c)
    segments, labels = zip(*c)

    return labels, segments

#Encode an input string given a codif    
def _encode(text, codif): 
    return text.encode(codif)

def checkConfusionMatrix(confusionMatrix):
    
    correctSum = 0
    correctTarg = 0
    incorrectSum = 0

    for i in range(len(confusionMatrix)):
        for j in range(len(confusionMatrix)):
            if i == j:
                correctSum += confusionMatrix[i][j]
                correctTarg += 1
            else:
                incorrectSum += confusionMatrix[i][j]

    return str(correctTarg), str(correctSum), str(incorrectSum)

def handleURLS(path_name): 
    return _loadSegments(_addrsFromPath(path_name))

def _loadSegments(addrs):
    
    ret_list = []

    for addr in addrs:
        with open(addr) as data_file:
            json_data = json.load(data_file)
            ret_list.append(json_data)
    
    return ret_list

def _loadDictionary(addr):
    
    with open(addr[0], 'r', encoding='utf-8') as data_file:
        return data_file.read()

def _generateLabels(addrs):
    
    labels = []

    for addr in addrs:
        if 'button' in addr.lower():
            labels.append(0)
        elif 'input' in addr.lower():
            labels.append(1)
        elif 'textarea' in addr.lower():
            labels.append(2)
        elif 'alert' in addr.lower():
            labels.append(3)
        elif 'table' in addr.lower():
            labels.append(4)    
        elif 'footer' in addr.lower():
            labels.append(5)
        elif 'link' in addr.lower():
            labels.append(6)
        elif 'sidebar' in addr.lower():
            labels.append(7)
        elif 'status' in addr.lower():
            labels.append(8)
        elif 'paragraph' in addr.lower():
            labels.append(9)
        elif 'br' in addr.lower():
            labels.append(10)
        elif 'timeline' in addr.lower():
            labels.append(11)
        elif 'item' in addr.lower():
            labels.append(12)
        elif 'header' in addr.lower():
            labels.append(13)
        else:
            labels.append(14)   
    
    return labels

def _replaceMultiple(string, toReplace, replacement):
    for item in toReplace:
        string = string.replace(item, replacement)

    return string

def _deleteTagsInNames(segment, percent):
    
    segment['name'] = segment['name'].lower()

    if random.randrange(0,100) > percent:
        segment['name'] = _replaceMultiple(segment['name'], labls, '')                        

#Generate a dictionary tuple from a given set of names
def _generateDictionary(words, name):   

    for index in range(len(words)):
        words[index] = '\''+words[index]+'\''

    words = sorted(words, key=str.lower)

    words = ','.join(words)

    with open(dicitonary_addr + name + '.txt', 'w', encoding='utf-8') as text_file:
        print('', file=text_file)
        print(f'{words}', file=text_file)

def _encodeStrings(obj):
    
    for key in obj:

        if isinstance(obj[key], str): 
            obj[key] = [_encode(obj[key].lower(), 'utf-8')]

        elif isinstance(obj[key], list):
            for subin in range(len(obj[key])):
                obj[key][subin] = _encode(obj[key][subin].lower(), 'utf-8')


#Split labels and segments given the % of train,val & test    
def _splitData(labels, segments, train_pr, validation_pr):
    
    train_labels = labels[0:int(train_pr*len(labels))]
    validation_labels = labels[int(train_pr*len(labels)):int(validation_pr*len(labels))]
    test_labels = labels[int(validation_pr*len(labels)):]

    train_segments = segments[0:int(train_pr*len(segments))]
    validation_segments = segments[int(train_pr*len(segments)):int(validation_pr*len(segments))]
    test_segments = segments[int(validation_pr*len(segments)):]

    train_tupla = {
        'segments': train_segments,
        'labels': train_labels
    }
    validation_tupla = {
        'segments': validation_segments,
        'labels': validation_labels
    }
    test_tupla = {
        'segments': test_segments,
        'labels': test_labels
    }

    return train_tupla, validation_tupla, test_tupla

#Consume an old file in a new one
def _consumeFile(old_file, new_file):
    old_text = open(old_file, 'r', encoding='utf-8').read()

    with open(old_file, 'w') as text_file:
                    print('', file=text_file)

    with open(new_file, 'w') as text_file:
                print('', file=text_file)
                print(f'{old_text}', file=text_file)

def _findWords(words, obj, keywords):
    
    for key in obj:
            if isinstance(obj[key], str) and key in keywords:
                obj[key] = re.sub(r'[^\s\w#-]+','',obj[key]) 
                obj[key] = obj[key].split(' ')
                for word in obj[key]:
                    words.append(word)

def _findWordsIn(words, obj):
    
    if isinstance(obj, str):
        obj = re.sub(r'[^\s\w#-]+','',obj) 
        obj = obj.split(' ')
        for word in obj:
            words.append(word)

def _countWords(words):
    
    tmp_words = []

    for word in words:
        word = re.sub(r'[^\w#-]+','',word) 
        tmp_words.append(word.lower())

    words_count = []
    new_words = []
    
    while len(tmp_words) > 0:
        count = 0
        
        if tmp_words[0] not in new_words:
            
            new_words.append(tmp_words[0])

            for i, w in reversed(list(enumerate(tmp_words))):
                
                if w in tmp_words[0]:
                    count += 1
                    del tmp_words[i]

            words_count.append(count)

    return new_words, words_count

def _parseDictionaryWords(words, minRep):

    tmp_words, count_words = _countWords(words)

    ret_words = []

    for index, tmp_word in enumerate(tmp_words):
        if count_words[index] >= minRep:
            ret_words.append(tmp_word)

    return ret_words

def _matchTags(segment, pattern):
    
    regex = re.compile(pattern)
    all = regex.findall(segment)

    return all


def handleData(paths_name, train_percent, val_percent, percent):
    
    train_pr = train_percent/100
    validation_pr = train_percent/100+val_percent/100

    print('Train %: ', train_pr*100,'\nValidation %: ', validation_pr*100-train_pr*100,'\nTest %: ', 100-validation_pr*100)

    addrs = _addrsFromPath(paths_name)

    segments = _loadSegments(addrs)
    labels = _generateLabels(addrs)

    labels, segments = _shuffleData(labels, segments)

    text_keywords = ['name', 'text', 'previous', 'next', 'parent']
    classes_keywords = ['class']
    text = []
    classes = []
    symbols = []
    colors = []

    for index in range(len(segments)):
        
        _deleteTagsInNames(segments[index], percent)

        _findWordsIn(colors, segments[index]['color'])
        _findWordsIn(symbols, segments[index]['symbolID'])
        _findWords(text, segments[index], text_keywords)
        _findWords(classes, segments[index]['frame'], classes_keywords)

        for i in range(0,5):
            if len(segments[index]['objects']) > i:
                _findWordsIn(text, segments[index]['objects'][i]['name'])
                _findWordsIn(classes, segments[index]['objects'][i]['class'])

        for i in range(0,5):
            obj = 'obj' + str(i+1)
            segments[index][obj] = {
                'name': 'None',
                'class': 'None'
            }

            if len(segments[index]['objects']) > (i):
                segments[index][obj] = {
                    'name': segments[index]['objects'][i]['name'],
                    'class': segments[index]['objects'][i]['class']
                }

        segments[index]['objects'] = ''

        _encodeStrings(segments[index])
        _encodeStrings(segments[index]['frame'])

        for i in range(1,6):
            obj = 'obj' + str(i)
            _encodeStrings(segments[index][obj])

    colors = _parseDictionaryWords(colors, minRep)
    text = _parseDictionaryWords(text, minRep)
    classes = _parseDictionaryWords(classes, minRep)
    symbols = _parseDictionaryWords(symbols, minRep)

    _generateDictionary(colors, 'colors')
    _generateDictionary(text, 'text')
    _generateDictionary(classes, 'classes')
    _generateDictionary(symbols, 'symbols')

    train_tupla, validation_tupla, test_tupla = _splitData(labels,
        segments,
        train_pr,
        validation_pr
        )

    for addr in addrs:
        new_addr = addr.replace(root_name, consumed_root_name)
        _consumeFile(addr, new_addr)
        os.remove(addr)

    return train_tupla, validation_tupla, test_tupla

def getDictionary(file_path):
    
    dictionary_addrs = _addrsFromPath(file_path)
    dictionary_txt = str(_loadDictionary(dictionary_addrs))
    dictionary_list = _matchTags(dictionary_txt, dictionary_pattern)
    dictionary_tuple = tuple(dictionary_list)

    return dictionary_tuple

def handleInferences(objs):

    ret = objs
    
    text_keywords = ['name', 'text', 'previous', 'next', 'parent']
    classes_keywords = ['class']

    for index, item in enumerate(ret):
        
        _ = []

        _findWordsIn(_, ret[index]['color'])
        _findWordsIn(_, ret[index]['symbolID'])
        _findWords(_, ret[index], text_keywords)
        _findWords(_, ret[index]['frame'], classes_keywords)

        for i in range(0,5):
            if len(ret[index]['objects']) > i:
                _findWordsIn(_, ret[index]['objects'][i]['name'])
                _findWordsIn(_, ret[index]['objects'][i]['class'])

        for i in range(0,5):
            obj = 'obj' + str(i+1)
            ret[index][obj] = {
                'name': 'None',
                'class': 'None'
            }

            if len(ret[index]['objects']) > (i):
                ret[index][obj] = {
                    'name': ret[index]['objects'][i]['name'],
                    'class': ret[index]['objects'][i]['class']
                }

        ret[index]['objects'] = ''

        _encodeStrings(ret[index])
        _encodeStrings(ret[index]['frame'])

        for i in range(1,6):
            obj = 'obj' + str(i)
            _encodeStrings(ret[index][obj])

    return ret