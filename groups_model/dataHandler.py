import re
import glob
import random
from pathlib import Path
import os
import json
import time

dictionary_pattern = "(?:\')+([0-9a-zA-Z]{1,})+(?:\')"
root_name = "dataset/groups"
consumed_root_name = "consumed/groups"
dicitonary_addr = "dataset/dictionaries/"
dictionary_tags = dicitonary_addr + 'tags.txt'

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
        if '' in addr.lower():
            labels.append(0)
    
    return labels

def _replaceMultiple(string, toReplace, replacement):
    for item in toReplace:
        string = string.replace(item, replacement)

    return string

def _deleteTagsInNames(segment, percent, _list):
    
    segment['name'] = segment['name'].lower()

    if random.randrange(0,100) > percent:
        segment['name'] = _replaceMultiple(segment['name'], _list, "")                        

#Generate a dictionary tuple from a given set of names
def _generateDictionary(words, name):   

    for index in range(len(words)):
        words[index] = words[index].replace("\n", "")
        words[index] = "'"+words[index]+"',"
        words[index] = words[index].lower()

    words = set(words)
    words = list(words)
    words = ''.join(words)

    with open(dicitonary_addr + name + '.txt', "w", encoding='utf-8') as text_file:
        print("", file=text_file)
        print(f"{words}", file=text_file)

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
        "segments": train_segments,
        "labels": train_labels
    }
    validation_tupla = {
        "segments": validation_segments,
        "labels": validation_labels
    }
    test_tupla = {
        "segments": test_segments,
        "labels": test_labels
    }

    return train_tupla, validation_tupla, test_tupla

#Consume an old file in a new one
def _consumeFile(old_file, new_file):
    old_text = open(old_file, 'r', encoding='utf-8').read()

    with open(old_file, "w") as text_file:
                    print("", file=text_file)

    with open(new_file, "w") as text_file:
                print("", file=text_file)
                print(f"{old_text}", file=text_file)

def _findWords(words, obj, keywords):
    
    for key in obj:
            if isinstance(obj[key], str) and key in keywords: 
                obj[key] = obj[key].split(" ")
                for word in obj[key]:
                    words.append(word)

def _findWordsIn(words, obj):
    
    if isinstance(obj, str): 
        obj = obj.split(" ")
        for word in obj:
            words.append(word)

def _matchTags(segment, pattern):
    
    regex = re.compile(pattern)
    all = regex.findall(segment)

    return all


def handleData(paths_name, train_percent, val_percent, percent):
    
    train_pr = train_percent/100
    validation_pr = train_percent/100+val_percent/100

    tag_list = getDictionaryList(dictionary_tags)

    print("Train %: ", train_pr*100,"\nValidation %: ", validation_pr*100-train_pr*100,"\nTest %: ", 100-validation_pr*100)

    addrs = _addrsFromPath(paths_name)

    segments = _loadSegments(addrs)
    labels = _generateLabels(addrs)

    print(segments)
    print(labels)

    labels, segments = _shuffleData(labels, segments)

    names_keywords = ['name', 'previous', 'next', 'tags']
    names = []

    for index in range(len(segments)):
        
        _deleteTagsInNames(segments[index], percent, tag_list)

        _findWords(names, segments[index], names_keywords)

        for tg in tag_list:
            segments[index][tg] = _countWords(segments[index]['tags'], tg)

        _encodeStrings(segments[index])

    _generateDictionary(names, 'names')

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
    
    dicitonary_addrs = _addrsFromPath(file_path)
    dictionary_txt = str(_loadDictionary(dicitonary_addrs))
    dictionary_list = _matchTags(dictionary_txt, dictionary_pattern)
    dictionary_set = set(dictionary_list)
    dictionary_list = list(dictionary_set)
    dictionary_tuple = tuple(dictionary_list)

    return dictionary_tuple

def getDictionaryList(file_path):
    
    dictionary_tuple = getDictionary(file_path)

    return list(dictionary_tuple)

def _countWords(segment, match):
    
    counter = 0
    for word in segment:
        if match == word:
            counter += 1
    
    return counter

