import numpy as np
import pickle as pk
import pandas as pd
from gensim.models.doc2vec import Doc2Vec

print("Loading data...")
data= pd.read_csv('cleaned_train.csv')


model= Doc2Vec.load("d2v.model")
vec= model.wv.syn0
print(vec.shape)
tags= model.docvecs.offset2doctag
N= len(tags)

train= []
sentiment= list(data['sent'])
for i in range(N):
    print(i)
    train.append([model.docvecs[str(i)],sentiment[i]/4])
with open('mat.pk','wb') as f:
    pk.dump(train,f)


