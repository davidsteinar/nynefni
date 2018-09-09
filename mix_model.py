import collections
import json
import glob

def normalize(counter):
    s = float(sum(counter.values()))
    for item,value in counter.items():
        counter[item] = value/s
    return counter
    
for filename in glob.iglob('data/names_count/*.txt'):
    print(filename)
    lang, model = filename.replace('data/names_count/','').replace('.txt','').split('_')
    with open('models/'+lang+'_'+model+'_2gram.json', 'r') as f:
        bigram = json.load(f)
    with open('models/'+lang+'_'+model+'_3gram.json', 'r') as f:
        trigram = json.load(f)

    #reduce bigram likelihood by 10
    minibigram = {}
    for key in bigram.keys():
        temp = {}
        for k,v in bigram[key].items():
            temp[k] = v*0.1
        minibigram[key] = temp
        
    mix = {}
    for trigram_key in trigram.keys():
        bigram_key = trigram_key[1:]
        combined = {**minibigram[bigram_key],**trigram[trigram_key]}
        normed = normalize(combined)
        mix[trigram_key] = normed
        
    with open('models/'+lang+'_'+model+'_23gram.json', 'w') as outfile:
        json.dump(mix, outfile)