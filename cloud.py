from wordcloud import WordCloud
import pandas as pd
import matplotlib.pyplot as plt

df= pd.read_csv('cleaned_djt.csv')

# print(df.info())

tweets= list(df['Text'])
text= " ".join(tweets)
# print(text[:5])
# print(type(text))

wc= WordCloud(width=1600,height=800,
    max_font_size=200,colormap='magma').generate(text)

plt.figure(figsize=(12,10))
plt.imshow(wc,interpolation='bilinear')
plt.axis("off")
plt.show()