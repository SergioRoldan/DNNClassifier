import base64
import binascii
import glob
import json
import os
import sys
from datetime import datetime

import requests
from PIL import Image

#Server urls
url_infer = 'http://localhost:7979/infer'
url_metrics = 'http://localhost:7979/metrics'


'''
Inputs: path (str)
Returns: address list 
'''


def _addrsFromPath(paths_name):
    return glob.glob(paths_name)


'''
Inputs: string to apply replace, list of substrings to be replaced, string replacement
Returns: the string replacing all toReplace with replacement
'''


def _replaceMultiple(string, toReplace, replacement):
    for item in toReplace:
        string = string.replace(item, replacement)

    return string


'''
Inputs: Instance object
Returns: The parsed instance
'''


def _parseSymbolInstance(instance):
    tmp_instance = {}

    # Replace empty field with None
    for key in instance:
        if instance[key] is '':
            tmp_instance[key] = 'None'
        else:
            tmp_instance[key] = instance[key]

    # Replace overwrite text with name if empty
    if len(instance['text']) == 0:
        tmp_instance['text'] = instance['name'].split('/')[0]

    # Define frame
    tmp_instance['frame'] = {
        'class': instance['frame']['<class>'],
        'height': instance['frame']['height'],
        'width': instance['frame']['width']
    }

    tmp_instance['identifier'] = binascii.b2a_hex(
        os.urandom(4)).decode('utf-8')

    return tmp_instance


'''
Inputs: One of the objects of the instance, the whole instance, the parent of the instance, the resulting symbol
Returns: The symbol modified
'''


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


'''
Inputs: a symbol instance, the parent of the symbol
Returns: a modified symbol
'''


def _createSymbol(instance, parent):

    symbol = {
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


'''
Inputs: group inside an artboard or other group, resulting instances, parent of the current group (defaults to None)
- Recursively look for all symbols or symbol instances in a group and its subgroups
'''


def _checkGroup(group, instances, parent=None):

    symbol = {}
    tmp_instances = []

    for groupOrInstance in group['layers']:
        print('\n                 GroupOrInstance Name:',
              groupOrInstance['name'])
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


'''
- Client main:
    Check and substitue command arguments, if any. Non-defined arguments are ignored
        If --infer looks for symbols inside a sketch json and page (if defined), parses the symbols and send them to the server to 
        infer, if non errors stores and display the results 
        If --metrics ask the server for the last metrics of the current serving model, if no errors stores and display the results
'''


if __name__ == '__main__':

    args = sys.argv[1:]

    if '-infer' in args:
        for arg in args:
            if '--file' in arg:
                path = arg.split('=')[1]
            if '--page-name' in arg:
                name = arg.split('=')[1]
                name = name.replace('*', ' ')
                print(name)

        if not 'path' in locals():
            print('File path argument missing')
            sys.exit('Write help or --help to see options')

        if not 'name' in locals():
            print('No page name arg, defaults to all')
            name = ''

        addrs = _addrsFromPath(path)
        instances = []

        for addr in addrs:
            with open(addr, encoding='utf8') as data_file:
                json_data = json.load(data_file)

            print('\n-- JSON: ', addr)

            for MSPage in json_data['pages']:

                if not name or 'all' in name:
                    print('\n Page name: ', MSPage['name'])
                    print(' - Class: ', MSPage['<class>'])
                    for MSArtboardGroup in MSPage['layers']:
                        print('\n         ArtboardGroup Name: ',
                              MSArtboardGroup['name'])
                        print('         - Class: ', MSArtboardGroup['<class>'])

                        if 'layers' in MSArtboardGroup:
                            _checkGroup(MSArtboardGroup, instances)

                elif name in MSPage['name']:
                    print('\n Page name: ', MSPage['name'])
                    print(' - Class: ', MSPage['<class>'])
                    for MSArtboardGroup in MSPage['layers']:
                        print('\n         ArtboardGroup Name: ',
                              MSArtboardGroup['name'])
                        print('         - Class: ', MSArtboardGroup['<class>'])

                        if 'layers' in MSArtboardGroup:
                            _checkGroup(MSArtboardGroup, instances)

        if len(instances) == 0:
            sys.exit('No objects could be find in the page or file specified')

        objects = [_parseSymbolInstance(group_ins) for group_ins in instances]
        inferences = {
            'inferences': objects
        }

        try:
            r = requests.post(url_infer, json=inferences)

            print('Status: ', r.status_code)

            if r.status_code != 200:
                sys.exit('Error' + r.text + ', try again later')

            res = eval(r.json())

            for item in res:
                print(item)

            time = _replaceMultiple(str(datetime.now()).split('.')[
                                    0], [' ', ':', '-'], '_')
            directory = 'metrics/'+time+'/'

            if not os.path.exists(directory):
                os.makedirs(directory)

            with open(directory+'inferences.txt', 'w') as json_file:
                json.dump(res, json_file)

        except Exception as e:
            sys.exit('Exception thrown during responde handle: '+str(e)+'. Try again later')

    elif '-metrics' in args:
        
        try:
            r = requests.get(url_metrics)

            print('Status: ', r.status_code)

            if r.status_code != 200:
                sys.exit('Error '+r.text+', try again later')

            time = _replaceMultiple(str(datetime.now()).split('.')[
                                    0], [' ', ':', '-'], '_')
            directory = 'metrics/'+time+'/'

            if not os.path.exists(directory):
                os.makedirs(directory)

            metrics = r.text.replace('\n', '')
            metrics = eval(metrics)

            imgdata = base64.b64decode(metrics['cm_pic'])
            filename = directory+'cm_pic.png'
            with open(filename, 'wb') as f:
                f.write(imgdata)

            cm_pic = Image.open(filename)
            cm_pic.show()

            imgdata = base64.b64decode(metrics['log_error_pic'])
            filename = directory+'log_error_pic.png'
            with open(filename, 'wb') as f:
                f.write(imgdata)

            log_error_pic = Image.open(filename)
            log_error_pic.show()

            cm = metrics['cm']
            with open(directory+'cm.txt', 'w') as text_file:
                print(f'{cm}', file=text_file)
            print('\nModel\'s confusion: ', cm)

            met = metrics['metrics']
            with open(directory+'metrics.txt', 'w') as text_file:
                print(f'{met}', file=text_file)
            print('Model\'s metrics: ', met)
        except Exception as e:
            sys.exit('Exception '+str(e)+'during metrics request')

    elif '--help' in args or  'help' in args:
        print('Arg: -infer: Infer an sketch page or document as json')
        print('         --file: File path to sketch json file')
        print('         --page-name (Optional): Page name to infere, defaults to all \n             Change spaces for ^ if any in the page name')
        print('Arg: -metrics: Ask server last training metrics')
        sys.exit()

    else:
        print('No correct args specified')
        sys.exit('Write -help to see option')
