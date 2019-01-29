import numpy as np
import pandas as pd
from keras.models import load_model
import matplotlib.pyplot as plt
import pickle as pk
from collections import Counter

print("Loading features...")
X= None
with open('testmat.pk','rb') as f:
    X= pk.load(f)

print("Loading database...")
df= pd.read_csv("cleaned_djt.csv")
df= df.drop(df.columns[df.columns.str.contains('unnamed',case=False)],axis=1)
months= [d.split(" ")[0][:-3] for d in list(df['Date'])]

print("Loading model...")
model= load_model("best_model.h5")

off= 200
off_= 1000
N= X.shape[0]
print("Predicitng sentiment...")
pred= model.predict(X)
pred-=np.mean(pred)
y= [np.mean(pred[i:i+off]) for i in range(0,N,off)]
y_= [np.mean(pred[i:i+off_]) for i in range(0,N,off_)]
x= np.array([i for i in range(len(y))])
x_= [int(np.mean(x[i:i+(off_/off)])) for i in range(0,len(x),(off_/off))]
dates_= [Counter(list(months[i:i+off_])).most_common(1)[0][0] for i in range(0,N,off_)]


plt.gca().invert_xaxis()
plt.scatter(x,y,s=10,label='Sliding window of 200 tweets')
plt.scatter(x_,y_,s=15,color='red',label='Sliding window of 1000 tweets')
plt.plot(x_,y_,color='red')
plt.xticks(x_,dates_)
plt.xticks(rotation=60)
plt.legend(loc= 'upper right')
plt.title("Donald Trump's tweets' Sentiment")
plt.xlabel("Year-Month")
plt.ylabel("Deviation from average Sentiment")
plt.show()

