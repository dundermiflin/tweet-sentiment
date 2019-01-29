import pandas as pd
import numpy as np
from gensim.models.doc2vec import Doc2Vec
import pickle as pk

print("Loading data...")
df= pd.read_csv('cleaned_djt.csv')
df= df.drop(df.columns[df.columns.str.contains('unnamed',case=False)],axis=1)
print(df.info())
tweets= list(df['Text'])

print("Loading embedding model...")
d2v= Doc2Vec.load("d2v.model")


test=[]
N= len(tweets)
for i in range(N):
    print("Generating embeddings for tweet {0}".format(i))
    terms= tweets[i].split(" ")
    vec= d2v.infer_vector(terms)
    test.append(vec)

test= np.stack(test,axis=0)
print(test.shape)

print("Saving embeddings...")
with open("testmat.pk",'wb') as f:
    pk.dump(test,f)