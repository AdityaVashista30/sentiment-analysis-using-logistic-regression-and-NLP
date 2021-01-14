import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfTransformer
import re
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegressionCV


df=pd.read_csv("data/movie_data.csv")
df.head(10)
df['review'][0]

tfidf=TfidfTransformer(use_idf=True,norm='l2',smooth_idf=True)
print(tfidf.fit_transform(bag).toarray())

df.loc[0,'review'][-50:]

def preprocessor(text):
    text = re.sub('<[^>]*>', '', text)
    emoticons = re.findall('(?::|;|=)(?:-)?(?:\)|\(|D|P)', text)
    text = re.sub('[\W]+', ' ', text.lower()) +        ' '.join(emoticons).replace('-', '')
    return text

preprocessor(df.loc[0,'review'][-50:])
df['review']=df['review'].apply(preprocessor)

from nltk.stem.porter import PorterStemmer
porter=PorterStemmer()

def tokenizer(text):
    return text.split()

def tokenizer_porter(text):
    return [porter.stem(word) for word in text.split()]

nltk.download('stopwords')

stop=stopwords.words('english')

tfidf=TfidfVectorizer(strip_accents=None,lowercase=False,preprocessor=None,tokenizer=tokenizer_porter,
                     use_idf=True,smooth_idf=True)
y=df.sentiment.values
x=tfidf.fit_transform(df.review)


x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=1,test_size=0.5)
clf=LogisticRegressionCV(cv=5,scoring='accuracy',random_state=0,n_jobs=-1,verbose=3,max_iter=300)
clf.fit(x_train,y_train)
pickle.dump(clf,open('saved_model.sav','wb'))

clf.score(x_test,y_test)

