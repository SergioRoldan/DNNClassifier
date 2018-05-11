import dataHandler as dh
import json
import os
import collections
import random as rand
import time
import re

#Get all the json addresses in dataset/JSONs
path = 'dataset/JSONs/*.json'
_classes = ['button', 'input', 'textarea', 'alert', 'table', 'header', 'timeline', 'paragraph','footer', 'br','link', 'menu', 'status', 'sidebar']

def _saveJSON(instance):
    instance['name'] = re.sub(r'[^\s\w#-]+','',instance['name'])
    file_name = 'dataset/symbols/' + instance['name'] + '_' + str(rand.randint(0,10**6)) + '.json'

    with open(file_name, 'w') as json_file:
        json.dump(instance, json_file)

def _createInstance(obj, instance, parent, symbol):

    if not symbol:
        symbol = {
            'name': instance['name'],
            'objects': [],
            'symbolID': '',
            'text': '',
            'color': '',
            'frame': instance['frame'],
            'previous': '',
            'next': '',
            'parent': ''
        }   

    if parent:
        symbol['parent'] = parent['name']

    if 'MSText' in obj['<class>']:
        symbol['text'] += obj['attributedString']['value']['text'] + ' '
        if symbol['color'] is '' and 'style' in obj:
            symbol['color'] = obj['style']['textStyle']['MSAttributedStringColorAttribute']['value']
    elif 'MSShape' in obj['<class>']:
        if 'style' in obj and len(obj['style']['fills']) > 0:
            symbol['color'] = obj['style']['fills'][0]['color']['value']

    symbol['objects'].append({
            'name': obj['name'],
            'class': obj['<class>']
    })
    
    return symbol

def _createSymbol(instance, parent):
    
    symbol =  {
        'name': instance['name'],
        'objects': [],
        'symbolID': instance['symbolID'],
        'text': '',
        'color': '',
        'frame': instance['frame'],
        'previous': '',
        'next': '',
        'parent': ''
    }

    if parent:
        symbol['parent'] = parent['name']

    if len(instance['overrideValues']) > 0:
        for val in instance['overrideValues']:
            symbol['text'] += val['value'] + ' '

    return symbol

def _checkGroup(group, instances, parent=None):

    symbol = {}
    tmp_instances = []

    for groupOrInstance in group['layers']:
        print('\n                 GroupOrInstance Name:', groupOrInstance['name'])
        print('                 - Class: ', groupOrInstance['<class>'])

        if 'LayerGroup' in groupOrInstance['<class>']:
            _checkGroup(groupOrInstance, instances, group)
        elif 'Symbol' in groupOrInstance['<class>']:
            tmp_instances.append(_createSymbol(groupOrInstance, parent))
        else:
            symbol = _createInstance(groupOrInstance, group, parent, symbol)

    if symbol:
        tmp_instances.append(symbol)

    for tmp_ins in tmp_instances:
        for index, ins in enumerate(reversed(instances)):
            if ins['parent'] in tmp_ins['parent'] and tmp_ins['parent'] != '': 
                list(reversed(instances))[index]['next'] = tmp_ins['name']
                tmp_ins['previous'] = ins['name']
                break

        instances.append(tmp_ins)

def generateRandomInstances(instanceObject):

    lst = []
    
    for _ in range(rand.randint(0,3)):
        
        tmpInsObj = instanceObject
        
        widthSign = bool(rand.randint(0,1))
        heightSign = bool(rand.randint(0,1))

        if widthSign and heightSign:
            tmpInsObj['frame']['width'] += rand.randint(0, int(tmpInsObj['frame']['width']/5))
            tmpInsObj['frame']['height'] += rand.randint(0,int(tmpInsObj['frame']['height']/5))
        elif heightSign and not widthSign:
            tmpInsObj['frame']['width'] -= rand.randint(0, int(tmpInsObj['frame']['width']/5))
            tmpInsObj['frame']['height'] += rand.randint(0, int(tmpInsObj['frame']['height']/5))
        elif not heightSign and widthSign:
            tmpInsObj['frame']['width'] += rand.randint(0, int(tmpInsObj['frame']['width']/5))
            tmpInsObj['frame']['height'] -= rand.randint(0, int(tmpInsObj['frame']['height']/5))
        else:
            tmpInsObj['frame']['width'] += rand.randint(0, int(tmpInsObj['frame']['width']/5))
            tmpInsObj['frame']['height'] -= rand.randint(0, int(tmpInsObj['frame']['height']/5))

        for key in tmpInsObj:
            
            if isinstance(tmpInsObj[key], str) and 'name' not in key:
                if rand.randint(0, 100) > 65:
                    tmpInsObj[key] = 'None'
            
            elif isinstance(tmpInsObj[key], list):
                for index, _ in enumerate(tmpInsObj[key]):
                    if rand.randint(0, 100) > 65:
                        tmpInsObj[key][index]['name'] = 'None'
                    if rand.randint(0, 100) > 65:
                        tmpInsObj[key][index]['class'] = 'None'
        
        lst.append(tmpInsObj)

    return lst

#Format a raw symbol instance into a parsed symbol instance
def _parseSymbolInstance(instance):
    tmp_instance = {}

    #Replace empty field with None
    for key in instance:
        if instance[key] is '':
            tmp_instance[key] = 'None'
        else:
            tmp_instance[key] = instance[key]

    #Replace overwrite text with name if empty
    if len(instance['text']) == 0:
        tmp_instance['text'] = instance['name'].split('/')[0]

    #Define frame
    tmp_instance['frame'] = {
        'class': instance['frame']['<class>'],
        'height': instance['frame']['height'], 
        'width': instance['frame']['width']
    }

    return tmp_instance   

def run():
    
    instances = []

    addrs = dh._addrsFromPath(path)
    
    if len(addrs) == 0:
        addrs = dh._addrsFromPath('tags_model/'+path)

    for addr in addrs:
        with open(addr, encoding='utf8') as data_file:
            json_data = json.load(data_file)

        print('\n-- JSON: ', addr)
        
        for MSPage in json_data['pages']:       
            print('\n Page name: ', MSPage['name'])
            print(' - Class: ',MSPage['<class>'])          
            for MSArtboardGroup in MSPage['layers']:
                print('\n         ArtboardGroup Name: ', MSArtboardGroup['name'])
                print('         - Class: ', MSArtboardGroup['<class>'])
                
                if 'layers' in MSArtboardGroup and 'Symbols' not in MSPage['name']:
                    
                    _checkGroup(MSArtboardGroup, instances)

    for group_ins in instances:

        if group_ins:

            tmp_instance = _parseSymbolInstance(group_ins)

            tmp_instances = generateRandomInstances(tmp_instance)
            tmp_instances.append(tmp_instance)

            for ins in tmp_instances:
                saved = False
                for _class in _classes:
                    if _class in ins['name'].lower():
                        _saveJSON(ins)
                        saved = True
                if not saved:
                    if rand.randrange(0,100) > 60:
                        _saveJSON(ins)
            
            

