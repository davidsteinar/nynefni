import collections
import json
import glob

def train_char_ngram(name_list, N):
    #given a list of names
    lm = collections.defaultdict(collections.Counter)
    for value in name_list:
        name, count = value.split(',')
        name_padded = ' '*N+name+' '
        for i in range(len(name_padded)-N):
            history, char = name_padded[i:i+N], name_padded[i+N]
            lm[history][char] += int(count)
    return lm

def normalize(counter):
    s = float(sum(counter.values()))
    for item,value in counter.items():
        counter[item] = value/s
    return counter

def norm_model(model):
    norm_model = collections.defaultdict(collections.Counter)
    for key in model:
        norm_model[key] = normalize(model[key])
    return norm_model

N = 3
for filename in glob.iglob('data/*.txt'):
    print(filename)
    with open(filename, 'r', encoding='utf-8') as f:
        name_list = [name for name in f.read().split()]
        
    trigram_model = train_char_ngram(name_list, N)
    trigram_model_norm = norm_model(trigram_model)
        
    model_location = 'models/'+filename.replace('data/','').replace('.txt','')+'_trigram.json'
    
    with open(model_location, 'w') as outfile:
        json.dump(trigram_model_norm, outfile)

