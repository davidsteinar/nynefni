import random
import collections
import json

class Nynefni:
    def __init__(self, language, category, N=3, unique=False):
        self.language = language
        self.category = category
        self.model_location = 'models/'+self.language+'_'+self.category+'_'+str(N)+'gram.json'
        self.constmodel = self.open_ngram_model(self.model_location)
        self.model = self.open_ngram_model(self.model_location)
        self.names = self.open_names('data/names/'+self.language+'_'+self.category+'.json') if unique else []
        self.N = 3 if N==23 else N

    def open_ngram_model(self, location):
        with open(location, 'r') as f:
            model = json.load(f)
        return model
    
    def open_names(self, location):
        with open(location, 'r', encoding='utf-8') as f:
            names = json.load(f)
        return names
        
    def mix_language(self, mix_language, category, weight):
        self.model = self.constmodel
        mix_names = self.open_names('data/names/'+mix_language+'_'+category+'.json')
        self.names = self.names + mix_names
        model_b_location = 'models/'+mix_language+'_'+category+'_'+str(self.N)+'gram.json'
        model_b = self.open_ngram_model(model_b_location)
        self.model = self.mix_models(model_b, self.model, weight)
    
    def generate_letter(self, key):
        key = key[-self.N:]
        letter_distribution = self.model[key]
        x = random.random()
        for c,v in letter_distribution.items():
            x = x - v
            if x <= 0:
                return c
  
    def generate_name(self,startkey='', maxlength=20):
        initkey = ' '*(self.N-len(startkey)) + startkey
        if not bool(self.model[initkey]):#if key does not exist
            return
        
        iteration = 0
        max_iterations = 100000
        
        while True:
            name = initkey
            key = initkey
            for i in range(maxlength):
                tmp = self.generate_letter(key)
                name = name + tmp
                key = name[-self.N:]
                if name[-1] == ' ' or i == maxlength-2:
                    name=name.strip()
                    break
            if name not in self.names:
                self.names.append(name.strip())
                return name.strip()
                
            iteration += 1
            if iteration > max_iterations:
                return 'No name was found'
        
    def norm_model(self,model):
        def normalize(counter):
            s = float(sum(counter.values()))
            for item,value in counter.items():
                counter[item] = value/s
            return counter
            
        norm_model = collections.defaultdict(collections.Counter)
        for key in model:
            norm_model[key] = normalize(model[key])
        return norm_model
    
    def mix_models(self,model_a, model_b, weight=1):
        mixmodel = collections.defaultdict(collections.Counter)
        for key in model_b.keys():
            for letter in model_b[key]:
                mixmodel[key][letter] = model_b[key][letter]
        for key in model_a.keys():
            for letter in model_a[key]:
                mixmodel[key][letter] = weight*model_a[key][letter]
        #now we renormalize the mixed model
        mixmodel_norm = self.norm_model(mixmodel)
        return mixmodel_norm
        
    def train_char_ngram(self, name_list):
        #given a list of names
        name_list = [' '*self.N+name+' ' for name in name_list]
        model = collections.defaultdict(collections.Counter)
        for name in name_list:
            for i in range(len(name)-self.N):
                history, char = name[i:i+self.N], name[i+self.N]
                model[history][char] += 1
        return self.norm_model(model)
    
    def influence_model(self, list_of_names, weight=5):
        self.model = self.constmodel
        influenced_model = self.train_char_ngram(list_of_names)
        #maybe edit letter distributions of unigrams
        self.model = self.mix_models(influenced_model, self.model, weight=weight)
        unigrams = ''.join(list_of_names)
        for uni in unigrams:
            for key in self.model:
                for letter in self.model[key]:
                    if self.model[key][uni] > 0:
                        self.model[key][uni] += 0.01
        self.model = self.norm_model(self.model)