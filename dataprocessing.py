#importing libraries 
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import chi2
import os

data_dir='C:/Users/HP/Desktop/stajprojesi/bbc'
print(os.listdir(data_dir))

"""
Reading data into Dataframes for easy overview of data and subsequent processing
"""
#creating list with matched categories 
from collections import defaultdict
frame = defaultdict(list)


for dirname, _, filenames in os.walk(data_dir):
    for filename in filenames:
        frame['category'].append(os.path.basename(dirname))
        
        name = os.path.splitext(filename)[0]
        frame['document_id'].append(name)

        path = os.path.join(dirname, filename)
        # throwed UnicodeDecodeError without encoding
        # Googled "UnicodeDecodeError: 'utf-8' codec can't decode byte 0xa3" - https://stackoverflow.com/a/55391198/7445772
        with open(path ,'r', encoding= 'unicode_escape') as file:
            frame['text'].append(file.read())

df = pd.DataFrame.from_dict(frame)
#print(df.head())

#print(df.tail())

#Exploring Data
print(df['category'].value_counts())
df.drop(0, axis=0, inplace=True)

print(len(df['document_id'].unique()))

#seems like incremental numbers for document names 
sorted_ids = sorted(df['document_id'].unique())
print(sorted_ids[:10], sorted_ids[-10:])

import random
num = 5
sample = random.sample(range(df.text.shape[0]), num)
"""
for idx in sample:
    print('*'*30)
    values = df.iloc[idx]
    print('Document ID : ', values['document_id'])
    print('Category : ', values['category'])
    print('Text : \n'+'-'*7)
    print(values['text'])
    print('='*36)
"""

#Data Ananlysis

"""
One of our main concerns when developing a classification model is whether
the different classes are balanced. This means
that the dataset contains an approximately equal portion of each class.
For example, if we had two classes and a 95% of observations belonging to one of them, 
a dumb classifier which always output the majority class would have 95% accuracy, although
it would fail all the predictions of the minority class

There are several ways of dealing with imbalanced datasets. 
One first approach is to undersample the majority class and oversample the minority one,
so as to obtain a more balanced dataset. Other approach can be using other error metrics
beyond accuracy such as the precision,
the recall or the F1-score.
"""
sns.countplot(df.category)
plt.title('Number of documents in each category')
#plt.show()


"""
We can see that the classes are approximately balanced,
so we won’t perform any undersampling or oversampling method.
However, we will anyway use precision and recall to evaluate model performance.
Another variable of interest can be the length of the news articles.
We can obtain the length distribution across categories:

"""


"""
On brief observation we can see that first line of text seems to be title and the next part is story/news article

We can split the text column into 2 separate features title and story
We can see there are financial symbols, punctuation marks, many stop words, new lines and some words like
 - "doesn't", "didn't" inside the text. These has to be taken care during preprocessing
"""

text = df["text"].str.split("\n", n = 1, expand = True) 

df["title"] =  text[0]
df['story'] = text[1]
#print(df.head())

category_story_word_count = defaultdict(list)
for category in df.category.unique():
    val = df[df['category']==category]['story'].str.split().apply(len).values
    category_story_word_count[category]=val
# distribution of stories across categories
plt.boxplot(category_story_word_count.values())
plt.title('Distribution of words in stories across categories')
keys = category_story_word_count.keys()
plt.xticks([i+1 for i in range(len(keys))], keys)
plt.ylabel('Words in stories')
plt.grid()
#plt.show()

# distribution of words in story
fig, axes = plt.subplots(2, 3, figsize=(15,8), sharey=True)
ax = axes.flatten()
plt.suptitle('Distribution of words in story')

for idx, (key, value) in enumerate(category_story_word_count.items()):
    sns.kdeplot(value,label=key, bw=0.6, ax=ax[idx])
plt.legend()
#plt.show()

"""


Preprocessing

Feature Engineering:

Feature engineering is the process of transforming data into features to act as inputs 
for machine learning models such that good quality features help in improving the model performance.
When dealing with text data, there are several ways of obtaining features that represent the data.
We will cover some of the most common methods and then choose the most suitable for our needs.

"""
"""
 Text cleaning
Before creating any feature from the raw text, we must perform a cleaning process to ensure no distortions are introduced to the model. We have followed these steps:
Special character cleaning: special characters such as “\n” double quotes must be removed from the text since we aren’t expecting any predicting power from them.

Upcase/downcase: we would expect, for example, “Book” and “book” to be the same word and have the same predicting power. For that reason we have downcased every word.

Punctuation signs: characters such as “?”, “!”, “;” have been removed.

Possessive pronouns: in addition, we would expect that “Trump” and “Trump’s” had the same predicting power.

Stemming or Lemmatization: stemming is the process of reducing derived words to their root. Lemmatization is the process of reducing a word to its lemma. The main difference between both methods is that lemmatization provides existing words, whereas stemming provides the root, which may not be an existing word. We have used a Lemmatizer based in WordNet.

Stop words: words such as “what” or “the” won’t have any predicting power since they will presumably be common to all the documents. For this reason, they may represent noise that can be eliminated. We have downloaded a list of English stop words from the nltk package and then deleted them from the corpus.
There is one important consideration that must be made at this point. We should take into account possible distortions that are not only present in the training test, but also in the news articles that will be scraped when running the web application.


"""
#Title - Text preprocessing
# printing random titles
samples = random.sample(range(len(df.title)), 10)
"""
for idx in samples:
    print(df.title[idx])
    print('-'*36)
"""
# stop words

nltk.download('stopwords')
stopwords = stopwords.words('english')
#print(stopwords)


#TEXT CLEANING 

import re

def clean_text(text):
    # decontraction : https://stackoverflow.com/a/47091490/7445772
    # specific
    text = re.sub(r"won't", "will not", text)
    text = re.sub(r"can\'t", "can not", text)
    # general
    text = re.sub(r"n\'t", " not", text)
    text = re.sub(r"\'re", " are", text)
    text = re.sub(r"\'s", " is", text)
    text = re.sub(r"\'d", " would", text)
    text = re.sub(r"\'ll", " will", text)
    text = re.sub(r"\'t", " not", text)
    text = re.sub(r"\'ve", " have", text)
    text = re.sub(r"\'m", " am", text)

    # remove line breaks \r \n \t remove from string 
    text = text.replace('\\r', ' ')
    text = text.replace('\\"', ' ')
    text = text.replace('\\t', ' ')
    text = text.replace('\\n', ' ')

    # remove stopwords
    text = ' '.join(word for word in text.split() if word not in stopwords)

    # remove special chars
    text = re.sub('[^A-Za-z0-9]+', ' ', text)
    #Upcase/downcase
    text = text.lower()
    return text

#Instantly make your loops show a smart progress meter - just wrap any iterable with tqdm(iterable),
#In addition to its low overhead, tqdm uses smart algorithms to predict the remaining time and to skip unnecessary iteration displays,
# # which allows for a negligible overhead in most cases.
from tqdm import tqdm

processed_titles = []
for title in tqdm(df['title'].values):
    processed_title = clean_text(title)
    processed_titles.append(processed_title)

# titles after processing
"""
print("**************Titles After Processing************************")
for idx in samples:
    print(processed_titles[idx])
    print('-'*36)
"""
#Story - Text preprocessing
"""for idx in samples[:2]:
    print(df.story[idx])
    print('-'*36)
"""
processed_stories = []
for story in tqdm(df['story'].values):
    processed_story = clean_text(story)
    processed_stories.append(processed_story)
"""
for i in range(3):
    print(df.category.values[i])
    print(processed_titles[i])
    print(processed_stories[i])
    print('-'*100)
"""
"""
**************************************************************
Preparing data for models
Feature engineering
*******************************************************
"""

df['title']=processed_titles
df['stor']=processed_story


#Label Coding 
#We'll create a dictionary with the label codification:
category_codes = {
    'business': 0,
    'entertainment': 1,
    'politics': 2,
    'sport': 3,
    'tech': 4
}
# Category mapping
df['Category_Code'] = df['category']
df = df.replace({'Category_Code':category_codes})

# Train - test split
"""
We'll set apart a test set to prove the quality of our models.
We'll do Cross Validation in the train set in order to tune the hyperparameters 
and then test performance on the unseen data of the test set.
"""
X_train, X_test, y_train, y_test = train_test_split(df['story'], df['Category_Code'], test_size=0.20, random_state=8)

"""
 Text representation
We have various options:

Count Vectors as features
TF-IDF Vectors as features
Word Embeddings as features
Text / NLP based features
Topic Models as features
We'll use TF-IDF Vectors as features.

We have to define the different parameters:

ngram_range: We want to consider both unigrams and bigrams.
max_df: When building the vocabulary ignore terms that have a document frequency strictly higher than the given threshold
min_df: When building the vocabulary ignore terms that have a document frequency strictly lower than the given threshold.
max_features: If not None, build a vocabulary that only consider the top max_features ordered by term frequency across the corpus.
"""
# Parameter election
ngram_range = (1,2)
min_df = 10
max_df = 1.
max_features = 300

tfidf = TfidfVectorizer(encoding='utf-8',
                        ngram_range=ngram_range,
                        stop_words=None,
                        lowercase=False,
                        max_df=max_df,
                        min_df=min_df,
                        max_features=max_features,
                        norm='l2',
                        sublinear_tf=True)
                        
features_train = tfidf.fit_transform(X_train).toarray()
labels_train = y_train
print(features_train.shape)

features_test = tfidf.transform(X_test).toarray()
labels_test = y_test
print(features_test.shape)

#We can use the Chi squared test in order to see what unigrams and bigrams are most correlated with each category:
from sklearn.feature_selection import chi2

for Product, category_id in sorted(category_codes.items()):
    features_chi2 = chi2(features_train, labels_train == category_id)
    indices = np.argsort(features_chi2[0])
    feature_names = np.array(tfidf.get_feature_names())[indices]
    unigrams = [v for v in feature_names if len(v.split(' ')) == 1]
    bigrams = [v for v in feature_names if len(v.split(' ')) == 2]
    """
    print("# '{}' category:".format(Product))
    print("  . Most correlated unigrams:\n. {}".format('\n. '.join(unigrams[-5:])))
    print("  . Most correlated bigrams:\n. {}".format('\n. '.join(bigrams[-2:])))
    print("")
"""
"""
# X_train
with open('Pickles/X_train.pickle', 'wb') as output:
    pickle.dump(X_train, output)
    """
# X_test    
with open('Pickles/X_test.pickle', 'wb') as output:
    pickle.dump(X_test, output)
    
# y_train
with open('Pickles/y_train.pickle', 'wb') as output:
    pickle.dump(y_train, output)
    
# y_test
with open('Pickles/y_test.pickle', 'wb') as output:
    pickle.dump(y_test, output)
    
# df
with open('Pickles/df.pickle', 'wb') as output:
    pickle.dump(df, output)
    
# features_train
with open('Pickles/features_train.pickle', 'wb') as output:
    pickle.dump(features_train, output)

# labels_train
with open('Pickles/labels_train.pickle', 'wb') as output:
    pickle.dump(labels_train, output)

# features_test
with open('Pickles/features_test.pickle', 'wb') as output:
    pickle.dump(features_test, output)

# labels_test
with open('Pickles/labels_test.pickle', 'wb') as output:
    pickle.dump(labels_test, output)
    
# TF-IDF object
with open('Pickles/tfidf.pickle', 'wb') as output:
    pickle.dump(tfidf, output)