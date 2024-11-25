#!/usr/bin/env python
# coding: utf-8

# In[113]:


get_ipython().system('pip install pyLDAvis')


# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re

import gensim
from gensim import corpora, models
import pyLDAvis
import pyLDAvis.gensim_models as gensimvis
from gensim.models import nmf
from gensim.models import CoherenceModel
#from gsdmm import MovieGroupProcess


# In[2]:


#Tweets=pd.read_csv('COVID-ARABIC-With-Emotion-all_month_and_stemmed.csv')
Tweets=pd.read_csv('COVID-ARABIC-With-Emotion-all_month_and_stemmed_new.csv')


# In[3]:


df=pd.read_csv('covid-arabic-all-month.csv')


# In[4]:


len(df)


# In[5]:


df['raw_text'][0]


# In[ ]:





# In[6]:


#Tweets


# In[7]:


word_to_check =  "كمام"
result = Tweets['raw_text'].str.contains(word_to_check, case=False)

print(result)


# In[ ]:


get_ipython().system('jupyter notebook --NotebookApp.iopub_data_rate_limit=10000000')


# In[ ]:





# In[23]:


for i in df['raw_text']:
    if 'عن بعد' in str(i):
        print(i)


# In[ ]:





# In[18]:


for i in df['raw_text']:
    if ' جديدة' in str(i):
        print(i)


# In[8]:


import pandas as pd
import nltk
from nltk import bigrams

# Sample DataFrame
 

# Define the word for which you want to find bigrams
target_word = 'اغلاق'

# Preprocess the text (e.g., lowercasing and tokenization)
#df['Text'] = df['Text'].apply(lambda x: x.lower())
def tokenize_text(text):
    if pd.notnull(text):
        return nltk.word_tokenize(text)
    else:
        return []

df['Tokens'] = df['raw_text'].apply(tokenize_text)

# Find bigrams containing the target word
target_bigrams = []
for tokens in df['Tokens']:
    bigram_list = list(bigrams(tokens))
    for bigram in bigram_list:
        if target_word in bigram:
            target_bigrams.append(' '.join(bigram))

print(target_bigrams)


# In[17]:


target_word = 'ايطاليا'


# In[18]:


target_bigrams = []
for tokens in df['Tokens']:
    bigram_list = list(bigrams(tokens))
    for bigram in bigram_list:
        if target_word in bigram:
            target_bigrams.append(' '.join(bigram))

print(target_bigrams)


# In[ ]:





# In[ ]:





# In[ ]:





# In[81]:


Tweets['raw_text'][0]


# In[5]:


Tweets['raw_text'] = Tweets['raw_text'].replace('فيروس', 'فايروس', regex=True)
#Tweets['raw_text'] = Tweets['raw_text'].replace('كورو', 'كورونا', regex=True)
#Tweets['raw_text'] = Tweets['raw_text'].replace('ون', '', regex=True)
#Tweets['raw_text'] = Tweets['raw_text'].replace('وفى', '', regex=True)
Tweets['raw_text'] = Tweets['raw_text'].replace('خلك', 'خليك', regex=True)
#Tweets['raw_text'] = Tweets['raw_text'].replace('ازم', 'الزم', regex=True)
#Tweets['raw_text'] = Tweets['raw_text'].replace('زم', 'الزم', regex=True)
Tweets['raw_text'] = Tweets['raw_text'].replace('حمدله', 'حمد', regex=True)


# In[ ]:





# In[ ]:





# In[6]:


texts = [[word for word in str(document).split()] for document in Tweets.raw_text ]


# In[7]:


docs = texts


# In[8]:


len(docs)


# In[9]:


dictionary_implemented = gensim.corpora.Dictionary(docs)


# In[10]:


bow_corpus = [dictionary_implemented.doc2bow(doc) for doc in docs]


# In[11]:


lda_model = gensim.models.LdaMulticore(bow_corpus, 
                                         num_topics= 16, 
                                         id2word= dictionary_implemented, 
                                         passes=10, iterations = 500, per_word_topics=True)


# In[91]:


len(docs)


# In[ ]:





# In[88]:


#replace   بلدي ببلد


# In[ ]:


فيروس  


# In[8]:


lda_model.print_topics(num_words=20)


# In[ ]:





# In[ ]:





# In[ ]:





# In[93]:


for i in docs:
    for w in i:
        if(w=="و"):
            print(i)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[24]:


for i in docs:
    for w in i:
        if(w=="وفي"):
            print(w)
        


# In[13]:


pyLDAvis.enable_notebook()
p = pyLDAvis.gensim_models.prepare(lda_model, bow_corpus, dictionary_implemented) 
p


# In[14]:


#len(corpus)


# In[14]:


from gensim.corpora import Dictionary


# In[15]:


dictionary = Dictionary(docs)


# In[16]:


bow_corpus = [dictionary_implemented.doc2bow(doc) for doc in docs]


# In[17]:


assigned_topics_new = []
for tweet in docs:
    topic_distribution = lda_model.get_document_topics(dictionary.doc2bow(tweet), per_word_topics=True)
    topic_probs = {topic_id: prob for topic_id, prob in topic_distribution[0]}
    assigned_topic = max(topic_probs, key=topic_probs.get)
    assigned_topics_new.append(assigned_topic)

Tweets['assigned_topic_new'] = assigned_topics_new


# In[18]:


import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.preprocessing import LabelEncoder


# In[19]:


Tweets['raw_text'] = Tweets['raw_text'].astype(str)


# Assuming you have a dataframe named 'df' with 'tweets' and 'emotion' columns

# Preprocessing (replace with your actual preprocessing function)
#df['clean_tweets'] = df['tweets'].apply(preprocess_function)

# Create text corpus
corpus = Tweets['raw_text'].tolist()

# Encode the emotion column
label_encoder = LabelEncoder()
Tweets['emotion_label'] = label_encoder.fit_transform(Tweets['predicted_emotion-bert'])


# In[ ]:





# In[ ]:





# In[ ]:


docs


# In[ ]:


Tweets['assigned_topic_new']


# In[ ]:


Tweets['assigned_topic_new']


# In[ ]:





# In[20]:


#Tweets['assigned_topic_new'] = topic_assignments.argmax(axis=1)

dominant_topics = Tweets.groupby('emotion_label')['assigned_topic_new'].value_counts().groupby(level=0).nlargest(1)
dominant_topics = dominant_topics.reset_index(level=0, drop=True)


# In[ ]:


dominant_topics


# In[21]:


topic_emotion_counts = Tweets.groupby(['assigned_topic_new', 'predicted_emotion-bert']).size().unstack().fillna(0)


# In[27]:


import matplotlib.pyplot as plt
import numpy as np

# Sample data (replace this with your actual data)
topic_emotion_counts = Tweets.groupby(['assigned_topic_new', 'predicted_emotion-bert']).size().unstack().fillna(0)

# Define emotions and corresponding colors
#emotions = ['joy', 'optimism', 'sadness', 'anger']  # Update with your desired emotions
#colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']  # Custom color palette

emotions = ['Anger', 'Disgust', 'Joy', 'Anticipation', 'Sadness', 'Surprise', 'Fear'] # Update with your desired emotions
colors = ['red', 'green', 'blue', 'orange','gray','brown','yellow']  # Update with your desired colors



# Plotting the number of tweets for each topic and emotion
plt.figure(figsize=(12,6))
topics = topic_emotion_counts.index
bar_width = 0.12
bar_padding = 0.0
index = np.arange(len(topics))





for i, emotion in enumerate(emotions):
    counts = topic_emotion_counts[emotion].values
    
    plt.bar(index + (i * bar_width) + (i * bar_padding), counts, bar_width, label=emotion, color=colors[i])

plt.xlabel('Topics')
plt.ylabel('Number of Tweets')
#plt.title('Number of Tweets per Topic by Emotion')
plt.legend(title='Emotions',title_fontsize=14, fontsize=12,loc='upper left', bbox_to_anchor=(1, 1))  # Legend outside the chart

#plt.legend()
plt.xticks(rotation=45, ha='right')
plt.xticks(index + (len(emotions) * bar_width + len(emotions) * bar_padding) / 2, topics)  # Adjust the position of xticks
plt.grid(axis='y', linestyle='--', alpha=0.7)  # Add grid lines for better readability

# Customize the legend
#legend = plt.legend(title='Emotions', title_fontsize=14, fontsize=12)
legend.get_title().set_color('black')  # Set legend title color

plt.tight_layout()
plt.show()


# In[21]:


import matplotlib.pyplot as plt
import numpy as np

# Sample data (replace this with your actual data)
# Emotion percentages by topic (update with your data)
topic_emotion_percentages = {
    'Topic': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
    'Anger (%)': [1.71, 2.55, 1.34, 1.94, 1.78, 2.80, 2.30, 1.21, 1.11, 2.02, 1.43, 1.85, 1.25, 1.87, 1.93, 1.43],
    'Anticipation (%)': [0.60, 0.70, 0.43, 0.77, 0.55, 1.04, 0.75, 0.49, 0.37, 0.68, 0.46, 0.61, 0.41, 0.59, 0.52, 0.49],
    'Disgust (%)': [1.49, 1.91, 1.05, 1.79, 1.54, 2.46, 1.98, 0.90, 0.84, 1.53, 1.08, 1.40, 0.97, 1.71, 1.73, 1.12],
    'Fear (%)': [0.29, 0.48, 0.25, 0.33, 0.20, 0.48, 0.31, 0.19, 0.19, 0.30, 0.19, 0.20, 0.23, 0.23, 0.31, 0.28],
    'Joy (%)': [1.48, 1.89, 1.00, 1.66, 1.25, 1.94, 1.93, 0.81, 0.81, 1.88, 0.93, 1.19, 0.83, 1.52, 2.06, 0.97],
    'Sadness (%)': [0.48, 0.50, 0.37, 0.49, 0.41, 0.81, 0.51, 0.32, 0.32, 0.49, 0.35, 0.31, 0.37, 0.35, 0.46, 0.43],
    'Surprise (%)': [0.28, 0.46, 0.24, 0.29, 0.36, 0.45, 0.37, 0.19, 0.19, 0.28, 0.23, 0.26, 0.24, 0.35, 0.33, 0.27]
}

# Create a DataFrame
df = pd.DataFrame(topic_emotion_percentages)

# Set the 'Topic' column as the index
df.set_index('Topic', inplace=True)

# Define emotions and corresponding colors
emotions = ['Anger (%)', 'Anticipation (%)', 'Disgust (%)', 'Fear (%)', 'Joy (%)', 'Sadness (%)', 'Surprise (%)']
colors = ['red', 'green', 'blue', 'orange', 'gray', 'brown', 'yellow']

# Plotting the percentage of emotions for each topic
plt.figure(figsize=(12, 6))
topics = df.index
bar_width = 0.12
bar_padding = 0.0
#index = np.arange(len(topics))
index = np.arange(len(df))

 
bottom = np.zeros(len(topics))

 
 

for i, emotion in enumerate(emotions):
    percentages = df[emotion].values
    plt.bar(topics + (i * bar_width)+(i * bar_padding) , percentages, width=bar_width, label=emotion, color=colors[i])

plt.xlabel('Topics')
plt.ylabel('Percentage (%)')
#plt.title('Percentage of Emotions in 16 Topics')




#plt.xticks(topics + bar_width * (len(emotions) / 2), topics)
#plt.legend(title='Emotions', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.legend(title='Emotions',title_fontsize=14, fontsize=12,loc='upper left', bbox_to_anchor=(1, 1))  # Legend outside the chart

plt.xticks(rotation=45, ha='right')
plt.xticks(index + (len(emotions) * bar_width + len(emotions) * bar_padding) / 2, topics)  # Adjust the position of xticks

plt.grid(axis='y', linestyle='--', alpha=0.7)
#legend.get_title().set_color('black')  # Set legend title color

plt.tight_layout()
plt.show()




 


# In[ ]:





# In[ ]:





# In[ ]:





# In[3]:


import matplotlib.pyplot as plt
import numpy as np

# Sample data (replace this with your actual data)
# Emotion percentages by topic (update with your data)
topic_emotion_percentages = {
    'Topic': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
    'Anger (%)': [1.71, 2.55, 1.34, 1.94, 1.78, 2.80, 2.30, 1.21, 1.11, 2.02, 1.43, 1.85, 1.25, 1.87, 1.93, 1.43],
    'Anticipation (%)': [0.60, 0.70, 0.43, 0.77, 0.55, 1.04, 0.75, 0.49, 0.37, 0.68, 0.46, 0.61, 0.41, 0.59, 0.52, 0.49],
    'Disgust (%)': [1.49, 1.91, 1.05, 1.79, 1.54, 2.46, 1.98, 0.90, 0.84, 1.53, 1.08, 1.40, 0.97, 1.71, 1.73, 1.12],
    'Fear (%)': [0.29, 0.48, 0.25, 0.33, 0.20, 0.48, 0.31, 0.19, 0.19, 0.30, 0.19, 0.20, 0.23, 0.23, 0.31, 0.28],
    'Joy (%)': [1.48, 1.89, 1.00, 1.66, 1.25, 1.94, 1.93, 0.81, 0.81, 1.88, 0.93, 1.19, 0.83, 1.52, 2.06, 0.97],
    'Sadness (%)': [0.48, 0.50, 0.37, 0.49, 0.41, 0.81, 0.51, 0.32, 0.32, 0.49, 0.35, 0.31, 0.37, 0.35, 0.46, 0.43],
    'Surprise (%)': [0.28, 0.46, 0.24, 0.29, 0.36, 0.45, 0.37, 0.19, 0.19, 0.28, 0.23, 0.26, 0.24, 0.35, 0.33, 0.27]
}

# Create a DataFrame
df = pd.DataFrame(topic_emotion_percentages)

# Set the 'Topic' column as the index
df.set_index('Topic', inplace=True)

# Define emotions and corresponding colors
emotions = ['Anger (%)', 'Disgust (%)', 'Joy (%)', 'Anticipation (%)', 'Sadness (%)', 'Surprise (%)', 'Fear (%)']
colors = ['red', 'green', 'blue', 'orange', 'gray', 'brown', 'yellow']

# Plotting the percentage of emotions for each topic
plt.figure(figsize=(12, 6))
topics = df.index
bar_width = 0.6

bottom = np.zeros(len(topics))

for emotion, color in zip(emotions, colors):
    percentages = df[emotion].values
    plt.bar(topics, percentages, width=bar_width, label=emotion, color=color, bottom=bottom)
    bottom += percentages

plt.xlabel('Topics')
plt.ylabel('Percentage (%)')
plt.title('Percentage of Emotions in 16 Topics')
plt.xticks(rotation=0)
plt.legend(title='Emotions', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(axis='y', linestyle='--', alpha=0.7)

plt.tight_layout()
plt.show()


# In[4]:


df


# In[13]:


import matplotlib.pyplot as plt
import numpy as np

# Sample data (replace this with your actual data)
# Emotion percentages by topic (update with your data)
topic_emotion_percentages = {
    'Topic': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
    'Anger (%)': [1.71, 2.55, 1.34, 1.94, 1.78, 2.80, 2.30, 1.21, 1.11, 2.02, 1.43, 1.85, 1.25, 1.87, 1.93, 1.43],
    'Anticipation (%)': [0.60, 0.70, 0.43, 0.77, 0.55, 1.04, 0.75, 0.49, 0.37, 0.68, 0.46, 0.61, 0.41, 0.59, 0.52, 0.49],
    'Disgust (%)': [1.49, 1.91, 1.05, 1.79, 1.54, 2.46, 1.98, 0.90, 0.84, 1.53, 1.08, 1.40, 0.97, 1.71, 1.73, 1.12],
    'Fear (%)': [0.29, 0.48, 0.25, 0.33, 0.20, 0.48, 0.31, 0.19, 0.19, 0.30, 0.19, 0.20, 0.23, 0.23, 0.31, 0.28],
    'Joy (%)': [1.48, 1.89, 1.00, 1.66, 1.25, 1.94, 1.93, 0.81, 0.81, 1.88, 0.93, 1.19, 0.83, 1.52, 2.06, 0.97],
    'Sadness (%)': [0.48, 0.50, 0.37, 0.49, 0.41, 0.81, 0.51, 0.32, 0.32, 0.49, 0.35, 0.31, 0.37, 0.35, 0.46, 0.43],
    'Surprise (%)': [0.28, 0.46, 0.24, 0.29, 0.36, 0.45, 0.37, 0.19, 0.19, 0.28, 0.23, 0.26, 0.24, 0.35, 0.33, 0.27]
}

# Create a DataFrame
df = pd.DataFrame(topic_emotion_percentages)

# Set the 'Topic' column as the index
df.set_index('Topic', inplace=True)

# Define emotions and corresponding colors
emotions = ['Anger (%)', 'Anticipation (%)', 'Disgust (%)', 'Fear (%)', 'Joy (%)', 'Sadness (%)', 'Surprise (%)']
colors = ['red', 'green', 'blue', 'orange', 'gray', 'brown', 'yellow']

# Plotting the percentage of emotions for each topic
plt.figure(figsize=(12, 6))
topics = df.index
bar_width = 0.12
bar_padding = 0.0
#index = np.arange(len(topics))
index = np.arange(len(df))

 
bottom = np.zeros(len(topics))


 

for i, emotion in enumerate(emotions):
    percentages = df[emotion].values
    plt.bar(topics + i * bar_width, percentages, width=bar_width, label=emotion, color=colors[i])

plt.xlabel('Topics')
plt.ylabel('Percentage (%)')
plt.title('Percentage of Emotions in 16 Topics')
plt.xticks(topics + bar_width * (len(emotions) / 2), topics)
plt.legend(title='Emotions', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(axis='y', linestyle='--', alpha=0.7)

plt.tight_layout()
plt.show()


# In[7]:


df


# In[ ]:





# In[83]:


counts


# In[87]:


for i, emotion in enumerate(emotions):
    print(i)
    counts = topic_emotion_counts[emotion].values
    print(sum(counts))


# In[86]:


10896 +9475  +9428  +3834 +3040 + 1757 +1835


# In[90]:


topic_emotion_counts


# In[97]:


def sum_of_rows(row):
  sum = 0
  
  sum += row
  return round((sum/637718)*100,2)

 
rows = [
  10896 + 3834 + 9475 + 1835 + 9428 + 3040 + 1757,
  16295 + 4434 + 12164 + 3079 + 12083 + 3217 + 2965,
  8559 + 2772 + 6729 + 1585 + 6366 + 2346 + 1554,
  12391 + 4894 + 11424 + 2110 + 10602 + 3113 + 1876,
  11345 + 3493 + 9838 + 1246 + 7969 + 2624 + 2295,
  17848 + 6630 + 15688 + 3078 + 12375 + 5192 + 2847,
  14702 + 4820 + 12601 + 1979 + 12329 + 3239 + 2342,
  7757 + 3130 + 5752 + 1224 + 5189 + 2031 + 1230,
  7082 + 2367 + 5344 + 1222 + 5168 + 2035 + 1228,
  12926 + 4321 + 9742 + 1896 + 11989 + 3132 + 1802,
  9113 + 2948 + 6901 + 1225 + 5953 + 2245 + 1486,
  11836 + 3915 + 9027 + 1288 + 7606 + 2015 + 1656,
  7988 + 2646 + 6286 + 1494 + 5282 + 2416 + 1547,
  11942 + 3801 + 10928 + 1437 + 9717 + 2231 + 2241,
  12328 + 3357 + 11007 + 1962 + 13172 + 2953 + 2124,
  9097 + 3087 + 7116 + 1779 + 6218 + 2762 + 1716
]

for i in rows:
    
    sum = sum_of_rows(i)

    print(sum)


# In[ ]:


40265
54237
29911
46410
38810
63658
52012
26313
24446
45808
29871
37343
27659
42297
46903
31775


# In[24]:


import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.ticker import FuncFormatter

# Sample data (replace this with your actual data)
#topic_emotion_counts = Tweets.groupby(['assigned_topic', 'Emotion']).size().unstack().fillna(0)

# Calculate percentage for each emotion within each topic

# Define emotions and corresponding colors
#emotions = ['joy', 'optimism', 'sadness', 'anger']  # Update with your desired emotions
#colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']  # Custom color palette


topic_emotion_counts = Tweets.groupby(['assigned_topic_new', 'predicted_emotion-bert']).size().unstack().fillna(0)
topic_emotion_percentages = topic_emotion_counts.div(topic_emotion_counts.sum(axis=1), axis=0) * 100


# Define emotions and corresponding colors
#emotions = ['joy', 'optimism', 'sadness', 'anger']  # Update with your desired emotions
#colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']  # Custom color palette

emotions = ['Anger', 'Disgust', 'Joy', 'Anticipation', 'Sadness', 'Surprise', 'Fear'] # Update with your desired emotions
colors = ['red', 'green', 'blue', 'orange','purple','pink','yellow']  # Update with your desired colors


# Function to format y-axis labels as percentages
def percentage(x, pos):
    return '{:.1f}%'.format(x)

# Plotting the percentage of tweets for each topic and emotion
plt.figure(figsize=(10,6))
topics = topic_emotion_percentages.index
bar_width = 0.12
bar_padding = 0.0
index = np.arange(len(topics))

 

for i, emotion in enumerate(emotions):
    percentages = topic_emotion_percentages[emotion].values
    plt.bar(index + (i * bar_width) + (i * bar_padding), percentages, bar_width, label=emotion, color=colors[i])

plt.xlabel('Topics')
plt.ylabel('Percentage')
#plt.title('Percentage of Tweets per Topic by Emotion')
plt.legend()
plt.xticks(rotation=45, ha='right')
plt.xticks(index + (len(emotions) * bar_width + len(emotions) * bar_padding) / 2, topics)  # Adjust the position of xticks
plt.grid(axis='y', linestyle='--', alpha=0.7)  # Add grid lines for better readability

# Set y-axis labels as percentages
formatter = FuncFormatter(percentage)
plt.gca().yaxis.set_major_formatter(formatter)

# Customize the legend
legend = plt.legend(title='Emotions', title_fontsize=14, fontsize=12)
legend.get_title().set_color('black')  # Set legend title color

plt.tight_layout()
plt.show()


# In[25]:


import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.ticker import FuncFormatter

# Sample data (replace this with your actual data)
#topic_emotion_counts = Tweets.groupby(['assigned_topic', 'Emotion']).size().unstack().fillna(0)

# Calculate percentage for each emotion within each topic

# Define emotions and corresponding colors
#emotions = ['joy', 'optimism', 'sadness', 'anger']  # Update with your desired emotions
#colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']  # Custom color palette


topic_emotion_counts = Tweets.groupby(['assigned_topic_new', 'predicted_emotion-bert']).size().unstack().fillna(0)
topic_emotion_percentages = topic_emotion_counts.div(topic_emotion_counts.sum(axis=1), axis=0) * 100


# Define emotions and corresponding colors
#emotions = ['joy', 'optimism', 'sadness', 'anger']  # Update with your desired emotions
#colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']  # Custom color palette

emotions = ['Anger', 'Disgust', 'Joy', 'Anticipation', 'Sadness', 'Surprise', 'Fear'] # Update with your desired emotions
colors = ['red', 'green', 'blue', 'orange','gray','brown','yellow']  # Update with your desired colors


# Function to format y-axis labels as percentages
def percentage(x, pos):
    return '{:.1f}%'.format(x)


plt.figure(figsize=(13, 6))
topics = topic_emotion_percentages.index
bar_width = 0.11
bar_padding = 0.0
index = np.arange(len(topics))

 

for i, emotion in enumerate(emotions):
    percentages = topic_emotion_percentages[emotion].values
    plt.bar(index + (i * bar_width) + (i * bar_padding), percentages, bar_width, label=emotion, color=colors[i])

plt.xlabel('Topics')
plt.ylabel('Percentage')
plt.legend(title='Emotions',title_fontsize=14, fontsize=12,loc='upper left', bbox_to_anchor=(1, 1))  # Legend outside the chart
plt.xticks(rotation=45, ha='right')
plt.xticks(index + (len(emotions) * bar_width + len(emotions) * bar_padding) / 2, topics)
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Set y-axis labels as percentages
formatter = FuncFormatter(percentage)
plt.gca().yaxis.set_major_formatter(formatter)

# Customize the legend
#legend = plt.legend(title='Emotions', title_fontsize=14, fontsize=12)
#legend.get_title().set_color('black')  # Set legend title color

plt.tight_layout()
plt.show()


# In[26]:


import matplotlib.pyplot as plt
import numpy as np

# Sample data (replace this with your actual data)
topic_emotion_counts = Tweets.groupby(['assigned_topic_new', 'predicted_emotion-bert']).size().unstack().fillna(0)

# Define emotions and corresponding colors
#emotions = ['joy', 'optimism', 'sadness', 'anger']  # Update with your desired emotions
#colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']  # Custom color palette

emotions = ['Anger', 'Disgust', 'Joy', 'Anticipation', 'Sadness', 'Surprise', 'Fear'] # Update with your desired emotions
colors = ['red', 'green', 'blue', 'orange','purple','pink','yellow']  # Update with your desired colors



# Plotting the number of tweets for each topic and emotion
plt.figure(figsize=(12, 8))
topics = topic_emotion_counts.index
bar_width = 0.12
bar_padding = 0.0
index = np.arange(len(topics))

for i, emotion in enumerate(emotions):
    counts = topic_emotion_counts[emotion].values
    plt.bar(index + (i * bar_width) + (i * bar_padding), counts, bar_width, label=emotion, color=colors[i])

plt.xlabel('Topics')
plt.ylabel('Number of Tweets')
#plt.title('Number of Tweets per Topic by Emotion')
plt.legend()
plt.xticks(rotation=45, ha='right')
plt.xticks(index + (len(emotions) * bar_width + len(emotions) * bar_padding) / 2, topics)  # Adjust the position of xticks
plt.grid(axis='y', linestyle='--', alpha=0.7)  # Add grid lines for better readability

# Customize the legend
legend = plt.legend(title='Emotions', title_fontsize=14, fontsize=12)
legend.get_title().set_color('black')  # Set legend title color

plt.tight_layout()
plt.show()


# In[ ]:





# In[28]:


topic_emotion_percentages


# In[66]:


sum([26.200593,8.843874,24.135375,4.466403,21.892292,8.542490,5.918972])


# In[46]:


docs


# In[70]:


import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from gensim.models import Phrases
from gensim.corpora import Dictionary
from gensim.models import TfidfModel
from gensim.models import LdaModel
from nltk.util import bigrams
import nltk
from nltk.corpus import brown
from nltk import FreqDist
from nltk import ngrams



# In[72]:


all_tweets = ' '.join(Tweets['raw_text'])

# Tokenize the text into individual words
tokens = word_tokenize(all_tweets)

# Find the most common unigrams and their frequencies
#unigram_freq = FreqDist(tokens)
#top_unigrams = unigram_freq.most_common(10)

# Calculate the percentage of each unigram in the dataset
#total_unigrams = len(tokens)
#unigram_percentage = [(word, count / total_unigrams * 100) for word, count in top_unigrams]

# Find the most common bigrams and their frequencies
bigrams = list(ngrams(tokens, 2))
bigram_freq = FreqDist(bigrams)
top_bigrams = bigram_freq.most_common(10)


# In[77]:


all_tokens = [token for doc in corpus_preprocessed for token in doc]
bigrams = list(ngrams(all_tokens, 2))

# Combine unigrams and bigrams
all_tokens_combined = all_tokens + [f"{bigram[0]}_{bigram[1]}" for bigram in bigrams]

# Create a dictionary and corpus (bag of words)
dictionary = Dictionary([all_tokens_combined])
corpus_bow = [dictionary.doc2bow(doc) for doc in corpus_preprocessed]



# In[80]:


all_tokens_combined


# In[ ]:





# In[55]:


#nltk.download('stopwords')
#nltk.download('punkt')

# Assuming 'corpus' is the list of your documents

# Preprocess and tokenize the corpus
def preprocess(text):
    stop_words = set(stopwords.words('english'))
    tokens = word_tokenize(text.lower())
    return [t for t in tokens if t not in stop_words]

corpus_preprocessed = [preprocess(doc) for doc in Tweets['raw_text']]


# In[69]:


corpus_preprocessed


# In[30]:


from nltk.collocations import BigramAssocMeasures, BigramCollocationFinder


# In[39]:


from gensim.models import Phrases


# In[74]:


#corpus_bigrams = [list(bigrams(doc)) for doc in corpus_preprocessed]
#corpus_with_bigrams = [[f"{bigram[0]}_{bigram[2]}" if isinstance(bigram, tuple) else word for word in doc] for doc in corpus_preprocessed]

# Create a Gensim dictionary
dictionary = Dictionary(bigram_freq)

# Convert the corpus to Gensim bag-of-words format
corpus_bow = [dictionary.doc2bow(doc) for doc in bigram_freq]



# In[75]:


corpus_bow


# In[ ]:





# In[ ]:


from gensim.corpora import Dictionary
from gensim.models import TfidfModel

# Create a Gensim dictionary
dictionary = Dictionary(corpus_bow)

# Convert the corpus to Gensim bag-of-words format
corpus_bow = [dictionary.doc2bow(doc) for doc in corpus_bow]

# Create a TF-IDF model
#tfidf = TfidfModel(corpus_bow)
#corpus_tfidf = tfidf[corpus_bow]


# In[ ]:





# In[78]:


from gensim.models import LdaModel

# Train the LDA model
num_topics = 16  # Adjust the number of topics based on your data
#lda_model = LdaModel(corpus_bow, id2word=dictionary, num_topics=num_topics)
lda_model = gensim.models.LdaMulticore(corpus_bow, 
                                         num_topics= 16, 
                                         id2word= dictionary, 
                                         passes=10, iterations = 500, per_word_topics=True)


# In[79]:


# View the topics
for idx, topic in lda_model.print_topics(num_words=20):
    print(f"Topic: {idx}")
    words = topic.split('+')
    for word in words:
        prob, word = word.strip().split('*')
        print(f"{word.strip()[1:-1]} (Prob: {prob.strip()})")
    print('\n')


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[49]:


#coh_list = list()

for i in range(25,26):
  print("="*100)
  print("for i = ",i)
  print("="*100)

  lda_model = gensim.models.LdaMulticore(bow_corpus,
                                         num_topics= i,
                                         id2word= dictionary_implemented,
                                         passes= 2, per_word_topics=True, iterations=100)

  coherence_model_lda = CoherenceModel(model=lda_model, texts=texts, dictionary=dictionary_implemented, coherence='c_npmi')
  coherence_lda = coherence_model_lda.get_coherence()
  print("coherence: ",coherence_lda)
  coh_list.append(coherence_lda)



# In[50]:


coh_list


# In[ ]:





# In[18]:


import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.preprocessing import LabelEncoder


# In[16]:


Tweets


# In[19]:


Tweets['raw_text'] = Tweets['raw_text'].astype(str)


# Assuming you have a dataframe named 'df' with 'tweets' and 'emotion' columns

# Preprocessing (replace with your actual preprocessing function)
#df['clean_tweets'] = df['tweets'].apply(preprocess_function)

# Create text corpus
corpus = Tweets['raw_text'].tolist()

# Encode the emotion column
label_encoder = LabelEncoder()
Tweets['emotion_label'] = label_encoder.fit_transform(Tweets['predicted_emotion-bert'])

# Vectorize the text data
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(corpus)

# Apply LDA topic modeling
num_topics = 16
# Set the number of desired topics
lda = LatentDirichletAllocation(n_components=num_topics, random_state=42)
lda.fit(X)

# Assign topics to tweets
topic_assignments = lda.transform(X)


# In[23]:


topic_assignments


# In[17]:


#num_topics = 16
# Set the number of desired topics
#lda = LatentDirichletAllocation(n_components=num_topics, random_state=42)
#lda.fit(X)

# Assign topics to tweets
#topic_assignments = lda.transform(X)


# In[ ]:





# In[28]:


Tweets


# In[19]:


#topic_assignments = lda_model.transform(X)


# In[29]:


#vectorizer = CountVectorizer()
#X = vectorizer.fit_transform(corpus)

# Apply LDA topic modeling
#num_topics = 16  # Set the number of desired topics
#lda = LatentDirichletAllocation(n_components=num_topics, random_state=42)
#lda.fit(X)

# Assign topics to tweets
#topic_assignments = lda.transform(X)


# In[11]:


feature_names = vectorizer.get_feature_names_out()

# Get the topics and their top words
num_top_words = 20  # Set the number of top words per topic
for topic_idx, topic in enumerate(lda.components_):
    top_words_idx = topic.argsort()[:-num_top_words - 1:-1]
    top_words = [feature_names[i] for i in top_words_idx]
    print(f"Topic {topic_idx + 1}:")
    print(", ".join(top_words))


# In[ ]:





# In[10]:


lda_model.print_topics(num_words=20)


# In[24]:


#Tweets['assigned_topic'] = topic_assignments.argmax(axis=1)


# In[25]:


#Tweets['assigned_topic'] = topic_assignments.argmax(axis=1)

# Calculate dominant topics based on emotion column
#Tweets['assigned_topic'] = topic_assignments.argmax(axis=1)

dominant_topics = Tweets.groupby('emotion_label')['assigned_topic'].value_counts().groupby(level=0).nlargest(1)
dominant_topics = dominant_topics.reset_index(level=0, drop=True)

# Decode emotion labels back to original values


# In[26]:


dominant_topics


# In[27]:


dominant_topics


# In[ ]:





# In[ ]:





# In[ ]:





# In[34]:


#topic_emotion_counts = merged_df.groupby(['assigned_topic', 'emotion']).size().unstack().fillna(0)

# Define emotions and corresponding colors
emotions = ['Anger', 'Disgust', 'Joy', 'Anticipation', 'Sadness', 'Surprise', 'Fear'] # Update with your desired emotions
colors = ['red', 'green', 'blue', 'orange','purple','pink','yellow']  # Update with your desired colors

 

# Plotting the number of tweets for each topic and emotion
plt.figure(figsize=(10,8))
topics = topic_emotion_counts.index
width = 0.45
bottom = np.zeros(len(topics+1))

for i, emotion in enumerate(emotions):
    counts = topic_emotion_counts[emotion].values
    plt.bar(topics, counts, width, bottom=bottom, label=emotion, color=colors[i])
    bottom += counts

plt.xlabel('Topics')
plt.ylabel('Number of Tweets')
#plt.title('Number of Tweets per Topic by Emotion')
plt.legend()
plt.xticks(rotation=45)
plt.xticks(np.arange(0, 16))

plt.show()


# In[9]:


Tweets


# In[53]:


import matplotlib.pyplot as plt
import numpy as np

# Sample data (replace this with your actual data)
topic_emotion_counts = Tweets.groupby(['assigned_topic_new', 'predicted_emotion-bert']).size().unstack().fillna(0)

# Define emotions and corresponding colors
#emotions = ['joy', 'optimism', 'sadness', 'anger']  # Update with your desired emotions
#colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']  # Custom color palette

emotions = ['Anger', 'Disgust', 'Joy', 'Anticipation', 'Sadness', 'Surprise', 'Fear'] # Update with your desired emotions
colors = ['red', 'green', 'blue', 'orange','purple','pink','yellow']  # Update with your desired colors



# Plotting the number of tweets for each topic and emotion
plt.figure(figsize=(12, 8))
topics = topic_emotion_counts.index
bar_width = 0.12
bar_padding = 0.0
index = np.arange(len(topics))

for i, emotion in enumerate(emotions):
    counts = topic_emotion_counts[emotion].values
    plt.bar(index + (i * bar_width) + (i * bar_padding), counts, bar_width, label=emotion, color=colors[i])

plt.xlabel('Topics')
plt.ylabel('Number of Tweets')
#plt.title('Number of Tweets per Topic by Emotion')
plt.legend()
plt.xticks(rotation=45, ha='right')
plt.xticks(index + (len(emotions) * bar_width + len(emotions) * bar_padding) / 2, topics)  # Adjust the position of xticks
plt.grid(axis='y', linestyle='--', alpha=0.7)  # Add grid lines for better readability

# Customize the legend
legend = plt.legend(title='Emotions', title_fontsize=14, fontsize=12)
legend.get_title().set_color('black')  # Set legend title color

plt.tight_layout()
plt.show()


# In[63]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Sample data (replace this with your actual data)
topic_emotion_counts = Tweets.groupby(['assigned_topic', 'predicted_emotion-bert']).size().unstack().fillna(0)

# Calculate the total number of tweets per topic
topic_total_tweets = topic_emotion_counts.sum(axis=1)

# Calculate the percentage of each emotion within each topic
topic_emotion_percentage = topic_emotion_counts.div(topic_total_tweets, axis=0) * 100

# Create a list to store the data rows
data_rows = []

# Populate the list with topic, emotion, and percentage data
for topic in topic_emotion_percentage.index:
    for emotion in topic_emotion_percentage.columns:
        percentage = topic_emotion_percentage.loc[topic, emotion]
        data_rows.append({'Topic': topic, 'Emotion': emotion, 'Percentage': percentage})

# Create a DataFrame from the list of data rows
df = pd.DataFrame(data_rows)

# Print the DataFrame
print(df)


# In[79]:


import matplotlib.pyplot as plt
import numpy as np

# Sample data (replace this with your actual data)
topic_emotion_counts = Tweets.groupby(['assigned_topic', 'predicted_emotion-bert']).size().unstack().fillna(0)

# Calculate percentage for each emotion within each topic
topic_emotion_percentages = topic_emotion_counts.div(topic_emotion_counts.sum(axis=1), axis=0) * 100

# Define emotions and corresponding colors
emotions = ['Anger', 'Disgust', 'Joy', 'Anticipation', 'Sadness', 'Surprise', 'Fear'] # Update with your desired emotions
colors = ['red', 'green', 'blue', 'orange','purple','pink','yellow']  # Update with your desired colors

# Function to format y-axis labels as percentages
def percentage(x, pos):
    return '{:.1f}%'.format(x)

# Plotting the percentage of tweets for each topic and emotion
plt.figure(figsize=(14,8))
topics = topic_emotion_percentages.index
bar_width = 0.14
bar_padding = 0.0
index = np.arange(len(topics))

for i, emotion in enumerate(emotions):
    percentages = topic_emotion_percentages[emotion].values
    plt.bar(index + (i * bar_width) + (i * bar_padding), percentages, bar_width, label=emotion, color=colors[i])

plt.xlabel('Topics')
plt.ylabel('Percentage')
#plt.title('Percentage of Tweets per Topic by Emotion')
plt.legend()
plt.xticks(rotation=45, ha='right')
plt.xticks(index + (len(emotions) * bar_width + len(emotions) * bar_padding) / 2, topics)  # Adjust the position of xticks
plt.grid(axis='y', linestyle='--', alpha=0.7)  # Add grid lines for better readability

# Set y-axis labels as percentages
formatter = FuncFormatter(percentage)
plt.gca().yaxis.set_major_formatter(formatter)

# Customize the legend
legend = plt.legend(title='Emotions', title_fontsize=14, fontsize=12)
legend.get_title().set_color('black')  # Set legend title color

plt.tight_layout()
plt.show()


# In[73]:


import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.ticker import FuncFormatter


# In[ ]:


import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.ticker import FuncFormatter

# Sample data (replace this with your actual data)
topic_emotion_counts = Tweets.groupby(['assigned_topic', 'Emotion']).size().unstack().fillna(0)

# Calculate percentage for each emotion within each topic
topic_emotion_percentages = topic_emotion_counts.div(topic_emotion_counts.sum(axis=1), axis=0) * 100

# Define emotions and corresponding colors
emotions = ['Anger', 'Disgust', 'Joy', 'Anticipation', 'Sadness', 'Surprise', 'Fear'] # Update with your desired emotions
colors = sns.color_palette("pastel", len(emotions))  # Prettier color palette from Seaborn

# Function to format y-axis labels as percentages
def percentage(x, pos):
    return '{:.1f}%'.format(x)

# Plotting the percentage of tweets for each topic and emotion
plt.figure(figsize=(10, 6))
topics = topic_emotion_percentages.index
bar_width = 0.15
bar_padding = 0.0
index = np.arange(len(topics))

for i, emotion in enumerate(emotions):
    percentages = topic_emotion_percentages[emotion].values
    plt.bar(index + (i * bar_width) + (i * bar_padding), percentages, bar_width, label=emotion, color=colors[i])

plt.xlabel('Topics')
plt.ylabel('Percentage')
#plt.title('Percentage of Tweets per Topic by Emotion')
plt.legend()
plt.xticks(rotation=45, ha='right')
plt.xticks(index + (len(emotions) * bar_width + len(emotions) * bar_padding) / 2, topics)  # Adjust the position of xticks
plt.grid(axis='y', linestyle='--', alpha=0.7)  # Add grid lines for better readability

# Set y-axis labels as percentages
formatter = FuncFormatter(percentage)
plt.gca().yaxis.set_major_formatter(formatter)

# Customize the legend
legend = plt.legend(title='Emotions', title_fontsize=14, fontsize=12)
legend.get_title().set_color('black')  # Set legend title color

plt.tight_layout()
plt.show()


# In[79]:


import matplotlib.pyplot as plt
import numpy as np

# Sample data (replace this with your actual data)
topic_emotion_counts = Tweets.groupby(['assigned_topic', 'Emotion']).size().unstack().fillna(0)

# Define emotions and corresponding colors
emotions = ['joy', 'optimism', 'sadness', 'anger']  # Update with your desired emotions
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']  # Custom color palette

# Plotting the number of tweets for each topic and emotion
plt.figure(figsize=(12, 8))
topics = topic_emotion_counts.index
bar_width = 0.12
bar_padding = 0.0
index = np.arange(len(topics))

for i, emotion in enumerate(emotions):
    counts = topic_emotion_counts[emotion].values
    plt.bar(index + (i * bar_width) + (i * bar_padding), counts, bar_width, label=emotion, color=colors[i])

plt.xlabel('Topics')
plt.ylabel('Number of Tweets')
#plt.title('Number of Tweets per Topic by Emotion')
plt.legend()
plt.xticks(rotation=45, ha='right')
plt.xticks(index + (len(emotions) * bar_width + len(emotions) * bar_padding) / 2, topics)  # Adjust the position of xticks
plt.grid(axis='y', linestyle='--', alpha=0.7)  # Add grid lines for better readability

# Customize the legend
legend = plt.legend(title='Emotions', title_fontsize=14, fontsize=12)
legend.get_title().set_color('black')  # Set legend title color

plt.tight_layout()
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




