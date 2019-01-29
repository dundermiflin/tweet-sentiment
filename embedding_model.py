import pandas as pd
import numpy as np
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import pickle as pk

df= pd.read_csv('cleaned_train.csv')
df= df.dropna(subset=['text'])
print(df.columns)
print(df.info())

tweets= list(df['text'])
docs= [TaggedDocument(words= d.split(" "),tags= [str(i)]) for i, d in enumerate(tweets)]


model= Doc2Vec(size= 400, alpha= 0.025,
                min_alpha=0.00025,min_count=1,dm=1)
model.build_vocab(docs)

for e in range(40):
    print("Epoch {0}".format(e))
    model.train(docs,total_examples=model.corpus_count,epochs= model.iter)
    model.alpha-=0.0002
    model.min_alpha= model.alpha

print("Saving model...")
model.save("d2v.model")
