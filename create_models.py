import collections
import json
import glob
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--N', type=int)
args = parser.parse_args()

def train_char_ngram(name_list, N):
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

for filename in glob.iglob('data/names_count/*.txt'):
    print(filename)
    with open(filename, 'r', encoding='utf-8') as f:
        name_list = [name for name in f.read().split()]
        
    Ngram_model = train_char_ngram(name_list, args.N)
    Ngram_model_norm = norm_model(Ngram_model)
        
    model_location = 'models/'+filename.replace('data/names_count/','').replace('.txt','')+'_'+str(args.N)+'gram.json'
    
    with open(model_location, 'w') as outfile:
        json.dump(Ngram_model_norm, outfile)



for filename in glob.iglob('data/names_count/*.txt'):
    print(filename)
    with open(filename, 'r', encoding='utf-8') as f:
        names=[]
        name_list = [name for name in f.read().split()]
        for value in name_list:
            name, count = value.split(',')
            names.append(name)
    
    rec = filename.replace('data/names_count/','').replace('.txt','')
    with open('data/names/'+rec+'.json', 'w') as outfile:
        json.dump(names, outfile)