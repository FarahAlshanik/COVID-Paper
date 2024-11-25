# -*- coding: utf-8 -*-
"""covid of bigrams and unigrams and their distribution.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1I5lHea9gxQItOTfPD_APlstlJwBBi0Km
"""

from google.colab import drive
drive.mount('/content/drive')

import locale
def getpreferredencoding(do_setlocale = True):
    return "UTF-8"
locale.getpreferredencoding = getpreferredencoding

!pip install bertopic[all]

!pip install flair

!pip install nltk

import pandas as pd
import numpy as np
#from bertopic import BERTopic
#from flair.embeddings import TransformerDocumentEmbeddings
#from gensim.models.coherencemodel import CoherenceModel
#import gensim.corpora as corpora
#from gensim.models import LdaMulticore
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF
import re

import matplotlib.pyplot as plt
import seaborn as sns
import string

path = ('/content/drive/MyDrive/Summer-Research/Topic Modeling/Final-Submission/TopicModeling-2023/')

ls

cd /content/drive/MyDrive

ls

import pandas as pd

tweets_df= pd.read_csv('COVID-ARABIC-With-Emotion-all_month_and_stemmed_new',encoding="utf-8")

tweets_df

tweets_df.info()

tweets_df = tweets_df.dropna(subset=['raw_text'])

from nltk.tokenize import word_tokenize

import nltk
from nltk.corpus import brown
from nltk import FreqDist
from nltk import ngrams

# Combine all the tweets into a single string
all_tweets = ' '.join(tweets_df['raw_text'])

# Tokenize the text into individual words
tokens = word_tokenize(all_tweets)

# Find the most common unigrams and their frequencies
unigram_freq = FreqDist(tokens)
top_unigrams = unigram_freq.most_common(10)

# Calculate the percentage of each unigram in the dataset
total_unigrams = len(tokens)
unigram_percentage = [(word, count / total_unigrams * 100) for word, count in top_unigrams]

# Find the most common bigrams and their frequencies
bigrams = list(ngrams(tokens, 2))
bigram_freq = FreqDist(bigrams)
top_bigrams = bigram_freq.most_common(10)

# Calculate the percentage of each bigram in the dataset
total_bigrams = len(bigrams)
bigram_percentage = [(word, count / total_bigrams * 100) for word, count in top_bigrams]

# Visualize the top unigrams using a word cloud
unigram_wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(unigram_freq)
plt.figure(figsize=(10, 5))
plt.imshow(unigram_wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title('Top Unigrams')
plt.show()

# Visualize the top bigrams using a word cloud
bigram_wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(bigram_freq)
plt.figure(figsize=(10, 5))
plt.imshow(bigram_wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title('Top Bigrams')
plt.show()

# Print the top unigrams and their percentages
print('Top Unigrams:')
for word, percentage in unigram_percentage:
    print(f'{word}: {percentage:.2f}%')

# Print the top bigrams and their percentages
print('\nTop Bigrams:')
for word, percentage in bigram_percentage:
    print(f'{word}: {percentage:.2f}%')

import nltk
nltk.download('punkt')

# Combine all the tweets into a single string
all_tweets = ' '.join(tweets_df['Text'])

# Tokenize the text into individual words
tokens = word_tokenize(all_tweets)

# Calculate the frequency of each word
word_freq = FreqDist(tokens)

# Calculate the total number of words
total_words = len(tokens)

# Find the most common unigrams and their frequencies
top_unigrams = word_freq.most_common(50)

# Calculate the percentage of each unigram in the dataset
unigram_percentage = [(word, count / total_words * 100) for word, count in top_unigrams]

# Create a table with the unigram and its percentage
unigram_table_data = []
for word, percentage in unigram_percentage:
    unigram_table_data.append([word, percentage])

# Display the table
unigram_table_df = pd.DataFrame(unigram_table_data, columns=['Unigram', 'Percentage'])
print("Unigram Table:")
print(unigram_table_df)

# Find the most common bigrams and their frequencies
bigrams = list(ngrams(tokens, 2))
bigram_freq = FreqDist(bigrams)
top_bigrams = bigram_freq.most_common(50)

# Calculate the percentage of each bigram in the dataset
bigram_percentage = [(word, count / len(bigrams) * 100) for word, count in top_bigrams]

# Create a table with the bigram and its percentage
bigram_table_data = []
for word, percentage in bigram_percentage:
    bigram_table_data.append([word, percentage])

# Display the table
bigram_table_df = pd.DataFrame(bigram_table_data, columns=['Bigram', 'Percentage'])
print("\nBigram Table:")
print(bigram_table_df)

# Create word clouds for unigrams and bigrams
unigram_wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(dict(top_unigrams))
bigram_wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(dict(top_bigrams))

# Display the word clouds
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(unigram_wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title('Unigram Word Cloud')

plt.subplot(1, 2, 2)
plt.imshow(bigram_wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title('Bigram Word Cloud')

plt.tight_layout()
plt.show()

# Tokenize the text into individual words
tokens = word_tokenize(all_tweets)

# Calculate the frequency of each word
word_freq = FreqDist(tokens)

# Calculate the total number of words
total_words = len(tokens)

# Find the most common unigrams and their frequencies
top_unigrams = word_freq.most_common(50)

# Calculate the percentage of each unigram in the dataset
unigram_percentage = [(word, count / total_words * 100) for word, count in top_unigrams]

# Create a table with the unigram and its percentage
unigram_table_data = []
for word, percentage in unigram_percentage:
    unigram_table_data.append([word, percentage])

# Display the table
unigram_table_df = pd.DataFrame(unigram_table_data, columns=['Unigram', 'Percentage'])
print("Unigram Table:")
print(unigram_table_df)

# Find the most common bigrams and their frequencies
bigrams = list(ngrams(tokens, 2))
bigram_freq = FreqDist(bigrams)
top_bigrams = bigram_freq.most_common(50)

# Calculate the percentage of each bigram in the dataset
bigram_percentage = [(word, count / len(bigrams) * 100) for word, count in top_bigrams]

# Create a table with the bigram and its percentage
bigram_table_data = []
for word, percentage in bigram_percentage:
    bigram_table_data.append([word, percentage])

# Display the table
bigram_table_df = pd.DataFrame(bigram_table_data, columns=['Bigram', 'Percentage'])
print("\nBigram Table:")
print(bigram_table_df)

# Convert bigrams to strings
top_bigrams_str = [' '.join(bigram) for bigram, count in top_bigrams]

# Create word clouds for unigrams and bigrams
unigram_wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(dict(top_unigrams))
bigram_wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_text(' '.join(top_bigrams_str))

# Display the word clouds
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(unigram_wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title('Unigram Word Cloud')

plt.subplot(1, 2, 2)
plt.imshow(bigram_wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title('Bigram Word Cloud')

plt.tight_layout()
plt.show()

# Tokenize the text into individual words
tokens = word_tokenize(all_tweets)

# Calculate the frequency of each word
word_freq = FreqDist(tokens)

# Calculate the total number of words
total_words = len(tokens)

# Find the most common unigrams and their frequencies
top_unigrams = word_freq.most_common(50)

# Calculate the percentage of each unigram in the dataset
unigram_percentage = [(word, count / total_words * 100) for word, count in top_unigrams]

# Create a table with the unigram and its percentage
unigram_table_data = []
for word, percentage in unigram_percentage:
    unigram_table_data.append([word, percentage])

# Display the table
unigram_table_df = pd.DataFrame(unigram_table_data, columns=['Unigram', 'Percentage'])
print("Unigram Table:")
print(unigram_table_df)

# Find the most common bigrams and their frequencies
bigrams = list(ngrams(tokens, 2))
bigram_freq = FreqDist(bigrams)
top_bigrams = bigram_freq.most_common(50)

# Calculate the percentage of each bigram in the dataset
bigram_percentage = [(word, count / len(bigrams) * 100) for word, count in top_bigrams]

# Create a table with the bigram and its percentage
bigram_table_data = []
for word, percentage in bigram_percentage:
    bigram_table_data.append([word, percentage])

# Display the table
bigram_table_df = pd.DataFrame(bigram_table_data, columns=['Bigram', 'Percentage'])
print("\nBigram Table:")
print(bigram_table_df)

# Convert bigrams to strings
top_bigrams_str = [' '.join(bigram) for bigram, count in top_bigrams]

# Create word clouds for unigrams and bigrams
unigram_wordcloud = WordCloud(width=800, height=400, background_color='white', font_path='ARABIC_FONT_PATH').generate_from_frequencies(dict(top_unigrams))
bigram_wordcloud = WordCloud(width=800, height=400, background_color='white', font_path='ARABIC_FONT_PATH').generate_from_text(' '.join(top_bigrams_str))

# Display the word clouds
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(unigram_wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title('Unigram Word Cloud')

plt.subplot(1, 2, 2)
plt.imshow(bigram_wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title('Bigram Word Cloud')

plt.tight_layout()
plt.show()

!pip install ar_wordcloud

from ar_wordcloud import ArabicWordCloud
awc = ArabicWordCloud(background_color="white")

t = 'أهلاً وسهلا، اللغة العربية جميلة'
wc = awc.from_text(unigram_wordcloud)

from ar_wordcloud import ArabicWordCloud


# Create word clouds for unigrams and bigrams
unigram_wordcloud = ArabicWordCloud(width=800, height=400, background_color='white').generate_from_frequencies(dict(top_unigrams))
bigram_wordcloud = ArabicWordCloud(width=800, height=400, background_color='white').generate_from_text(' '.join(top_bigrams_str))

# Display the word clouds
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.imshow(unigram_wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title('Top Unigrams Word Cloud')

plt.subplot(1, 2, 2)
plt.imshow(bigram_wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title('Top Bigrams Word Cloud')

plt.tight_layout()
plt.show()

# Display the word clouds
import matplotlib.pyplot as plt

plt.figure(figsize=(20, 6))

plt.subplot(1, 2, 1)
plt.imshow(unigram_wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title('Top Unigrams Word Cloud')

# Display the word clouds
import matplotlib.pyplot as plt

plt.figure(figsize=(15, 6))

plt.subplot(1, 2, 2)
plt.imshow(bigram_wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title('Top Bigrams Word Cloud')

plt.tight_layout()
plt.show()

import matplotlib.pyplot as plt
from wordcloud import WordCloud

# Word cloud for unigrams
plt.figure(figsize=(15, 6))


# Word cloud for bigrams
plt.subplot(1, 2, 2)
plt.imshow(bigram_wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title('Top Bigrams Word Cloud')



# Use the same color for bigram word cloud
bigram_wordcloud.recolor(color_func=lambda *args, **kwargs: 'black')

plt.tight_layout()
plt.show()

import matplotlib.pyplot as plt
from wordcloud import WordCloud

plt.figure(figsize=(15, 6))

plt.subplot(1, 2, 2)

# Define a function that returns a constant color
def color_func(word, font_size, position, orientation, random_state, **kwargs):
    return 'black'

# Create the bigram word cloud with the color_func parameter set
bigram_wordcloud = WordCloud(color_func=color_func, interpolation='bilinear')

# Generate the word cloud
bigram_wordcloud.generate_from_frequencies(bigram_frequencies)

# Display the bigram word cloud
plt.imshow(bigram_wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title('Top Bigrams Word Cloud')

plt.tight_layout()
plt.show()

import matplotlib.pyplot as plt

plt.figure(figsize=(15, 6))

plt.subplot(1, 2, 2)

# Define a function that returns a constant color
def color_func(word, **kwargs):
    return 'black'

# Create the ArabicWordCloud object with the color_func parameter set
arabic_wordcloud = ArabicWordCloud(color_func=color_func, font_path='path/to/your/font')

# Generate the word cloud
arabic_wordcloud.generate_from_frequencies(bigram_frequencies)

# Display the word cloud
plt.imshow(arabic_wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title('Top Bigrams Word Cloud')

plt.tight_layout()
plt.show()

import pandas as pd
import matplotlib.pyplot as plt
from nltk import bigrams
from wordcloud import STOPWORDS

# Assuming you have a DataFrame called 'tweets_df' with a column 'text' containing the tweets

# Preprocess the tweets and extract bigram words
bigram_words = []
for tweet in tweets_df['raw_text']:
    words = tweet.split()  # Split the tweet into individual words
    bigrams_list = list(bigrams(words))  # Generate bigrams from the words
    bigram_words.extend([' '.join(bigram) for bigram in bigrams_list])

# Calculate the frequency of each bigram word
bigram_word_freq = {}
for word in bigram_words:
    if word in bigram_word_freq:
        bigram_word_freq[word] += 1
    else:
        bigram_word_freq[word] = 1

# Create the ArabicWordCloud object
arabic_wordcloud = ArabicWordCloud()

# Generate the word cloud from the bigram word frequencies
arabic_wordcloud.generate_from_frequencies(bigram_word_freq)

# Plot the word cloud
plt.figure(figsize=(10, 6))
plt.imshow(arabic_wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title('Bigram Arabic Words Word Cloud')
plt.show()

import pandas as pd
import matplotlib.pyplot as plt
from nltk import bigrams
from wordcloud import STOPWORDS

# Assuming you have a DataFrame called 'tweets_df' with a column 'text' containing the tweets

# Preprocess the tweets and extract bigram words
bigram_words = []
for tweet in tweets_df['Text']:
    words = tweet.split()  # Split the tweet into individual words
    bigrams_list = list(bigrams(words))  # Generate bigrams from the words
    bigram_words.extend([' '.join(bigram) for bigram in bigrams_list])

# Calculate the frequency of each bigram word
bigram_word_freq = {}
for word in bigram_words:
    if word in bigram_word_freq:
        bigram_word_freq[word] += 1
    else:
        bigram_word_freq[word] = 1

# Create the ArabicWordCloud object
arabic_wordcloud = ArabicWordCloud()

# Generate the word cloud from the bigram word frequencies
arabic_wordcloud.generate_from_frequencies(bigram_word_freq)

# Plot the word cloud
plt.figure(figsize=(10, 6))
plt.imshow(arabic_wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title('Bigram Arabic Words Word Cloud')
plt.show()

bigram_words = []
for tweet in tweets_df['raw_text']:
    words = tweet.split()  # Split the tweet into individual words
    bigrams_list = list(bigrams(words))  # Generate bigrams from the words
    bigram_words.extend([' '.join(bigram) for bigram in bigrams_list])

# Calculate the frequency of each bigram word
bigram_word_freq = {}
for word in bigram_words:
    if word in bigram_word_freq:
        bigram_word_freq[word] += 1
    else:
        bigram_word_freq[word] = 1

# Create the ArabicWordCloud object
arabic_wordcloud = ArabicWordCloud(background_color='white')

# Generate the word cloud from the bigram word frequencies
arabic_wordcloud.generate_from_frequencies(bigram_word_freq)

# Plot the word cloud
plt.figure(figsize=(15, 8))
plt.imshow(arabic_wordcloud, interpolation='bilinear')
plt.axis('off')
#plt.title('Bigram  Words Word Cloud')
plt.show()

import matplotlib.pyplot as plt
from wordcloud import STOPWORDS

# Assuming you have a DataFrame called 'tweets_df' with a column 'text' containing the tweets

# Preprocess the tweets and extract unigram words
unigram_words = []
for tweet in tweets_df['raw_text']:
    words = tweet.split()  # Split the tweet into individual words
    unigram_words.extend(words)

# Calculate the frequency of each unigram word
unigram_word_freq = {}
for word in unigram_words:
    if word in unigram_word_freq:
        unigram_word_freq[word] += 1
    else:
        unigram_word_freq[word] = 1

# Create the ArabicWordCloud object
arabic_wordcloud = ArabicWordCloud(background_color='white')

# Generate the word cloud from the unigram word frequencies
arabic_wordcloud.generate_from_frequencies(unigram_word_freq)

# Plot the word cloud
plt.figure(figsize=(15, 8))
plt.imshow(arabic_wordcloud, interpolation='bilinear')
plt.axis('off')
#plt.title('Unigram Arabic Words Word Cloud')
plt.show()


