import torch
import torch.nn as nn
import torch.functional as F
import torch.optim as optim
import pandas as pd
import numpy as np
import pickle as pk
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

learning_rate=0.01
epochs=1

class Net(nn.Module):

    def __init__(self):
        super(Net,self).__init__()
        self.fc1= nn.Linear(30,64)
        self.fc2= nn.Linear(64,32)
        self.fc3= nn.Linear(32,1)

    def forward(self, x):
        x= torch.sigmoid(self.fc1(x))
        x= torch.sigmoid(self.fc2(x))
        x= torch.sigmoid(self.fc3(x))
        return x


print("Loading data...")
data= None
with open('feat_hash.pk','rb') as f:
    data= pk.load(f)
data= np.array(data)
X= np.stack(data[:,0],axis=0)
y= data[:,1]/4
y= y.astype(int)
X_train,X_test,y_train,y_test= train_test_split(X,y,test_size=0.1)
X_train= torch.from_numpy(X_train)
y_train= torch.from_numpy(y_train)
X_test= torch.from_numpy(X_test)
y_test= torch.from_numpy(y_test)

print("Building model...")
net= Net()
net.double()
criterion= nn.MSELoss()
optimizer= optim.Adam(net.parameters(),lr=learning_rate)
# print(net)

print("Training model...")
for e in range(epochs):
    epoch_loss=0
    for i in range(y_train.shape[0]):
        optimizer.zero_grad()
        out= net(X_train[i])
        loss= criterion(out, y_train[i])
        loss.backward()
        optimizer.step()
        epoch_loss+=loss.item()
    print("Epoch:{0} loss:{1} ".format(e,epoch_loss))
