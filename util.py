import pandas as pd

df= pd.read_csv('training.csv',header=None)

df.drop(df.columns[[1,2,3,4]],axis=1,inplace=True)
df1= df[:100000]
df2= df[800000:900000]
df3= pd.concat([df1,df2],axis=0)
print(df3.info())
df3.to_csv('truncated.csv',header= True,index=None)
