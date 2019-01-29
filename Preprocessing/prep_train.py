import pandas as pd
from HTMLParser import HTMLParser
import re
from nltk.stem import SnowballStemmer
from nltk.tokenize import WordPunctTokenizer
import sys
reload(sys)
sys.setdefaultencoding('utf-8')

parser= HTMLParser()
sno= SnowballStemmer('english')
tok= WordPunctTokenizer()
negations_dic = {"isn't":"is not", "aren't":"are not", "wasn't":"was not", "weren't":"were not",
                "haven't":"have not","hasn't":"has not","hadn't":"had not","won't":"will not",
                "wouldn't":"would not", "don't":"do not", "doesn't":"does not","didn't":"did not",
                "can't":"can not","couldn't":"could not","shouldn't":"should not","mightn't":"might not",
                "mustn't":"must not"}
df= pd.read_csv('truncated.csv')
print(df.info())
print(df.columns)
N= 200000

cleaned= pd.DataFrame()

for i in range(N):
    print("Cleaning tweet {0}".format(i))
    tweet= df.iloc[i]['5']
    val= df.iloc[i]['0']
    tweet= parser.unescape(tweet)
    tweet= re.sub(r'@[^\s]+','',tweet)
    tweet= re.sub(r'https?://(www.)?[^\s]+','',tweet)
    tweet= re.sub(r'www.[^\s]+','',tweet)
    tweet= re.sub(r'pic.twitter[^\s]+','',tweet)
    tweet= re.sub(r'#','',tweet)
    tweet= tweet.lower()
    words= tok.tokenize(tweet)
    for j in range(len(words)):
        if words[j] in negations_dic.keys():
            # print(words[j])
            words[j]= negations_dic[words[j]]
    tweet= " ".join(words)
    tweet= re.sub(r'[^A-Za-z]'," ",tweet)
    words= tok.tokenize(tweet)
    terms= [sno.stem(w) for w in words]
    tweet= " ".join(terms).strip()
    cleaned= cleaned.append({'text':tweet,'sent':val},ignore_index=True)
    print(tweet)

cleaned.to_csv('cleaned_train.csv',index=None)
    
