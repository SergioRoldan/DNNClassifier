import glob
import itertools
import json
import os
import random
import re
import time
from pathlib import Path


# Parameters to train, evaluate and predict the model
dictionary_pattern = r'(?:\')+([\w#-]+)+(?:\')'
dicitonary_addr = 'dataset/dictionaries/'


'''
Inputs: path (str)
Returns: address list 
'''


def _addrsFromPath(paths_name):
    return glob.glob(paths_name)


'''
    Inputs: confusion matrix
    Returns: number of targets predicted, correct predictions confidence, incorrect predictions confidence
'''


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


'''
Inputs: text and codification (strs)
Returns: text encoded with codif
'''


def _encode(text, codif):
    return text.encode(codif)


'''
Inputs: object
- Encodes utf-8 an object
'''


def _encodeStrings(obj):

    for key in obj:

        if isinstance(obj[key], str):
            obj[key] = [_encode(obj[key].lower(), 'utf-8')]

        elif isinstance(obj[key], list):
            for subin in range(len(obj[key])):
                obj[key][subin] = _encode(obj[key][subin].lower(), 'utf-8')


'''
Inputs: string to apply replace, list of substrings to be replaced, string replacement
Returns: the string replacing all toReplace with replacement
'''


def _replaceMultiple(string, toReplace, replacement):
    for item in toReplace:
        string = string.replace(item, replacement)

    return string


'''
Inputs: words list, object, keywords list
- Find words in an object's strings which key are in keywords list
Note: object's strings are splitted and any EOF or whitespace eliminated
'''


def _findWords(words, obj, keywords):

    for key in obj:
        if isinstance(obj[key], str) and key in keywords:
            obj[key] = re.sub(r'[^\s\w#-]+', '', obj[key])
            obj[key] = obj[key].split(' ')
            for word in obj[key]:
                words.append(word)


'''
Inputs: words list, object
- Find words in an object's strings
Note: object's strings are splitted and any EOF or whitespace eliminated
'''


def _findWordsIn(words, obj):

    if isinstance(obj, str):
        obj = re.sub(r'[^\s\w#-]+', '', obj)
        obj = obj.split(' ')
        for word in obj:
            words.append(word)


'''
Inputs: segment (str), pattern (regex)
Returns: the matched sequences of segment according to pattern
'''


def _matchTags(segment, pattern):

    regex = re.compile(pattern)
    all = regex.findall(segment)

    return all


'''
Inputs: a dictionary address
Returns: the data stored in the dictionary
'''


def _loadDictionary(addr):

    with open(addr[0], 'r', encoding='utf-8') as data_file:
        return data_file.read()


'''
Inputs: a dictionary path
Returns: a dictionary tuple parsed
'''


def getDictionary(file_path):

    dictionary_addrs = _addrsFromPath(file_path)
    dictionary_txt = str(_loadDictionary(dictionary_addrs))
    dictionary_list = _matchTags(dictionary_txt, dictionary_pattern)
    dictionary_tuple = tuple(dictionary_list)

    return dictionary_tuple


'''
Inputs: list of objects
Returns: list of objects parsed to be stored
'''


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

        for i in range(0, 5):
            if len(ret[index]['objects']) > i:
                _findWordsIn(_, ret[index]['objects'][i]['name'])
                _findWordsIn(_, ret[index]['objects'][i]['class'])

        for i in range(0, 5):
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

        for i in range(1, 6):
            obj = 'obj' + str(i)
            _encodeStrings(ret[index][obj])

    return ret
