#!pip install spacy
#!pip install newsapi-python
#!python -m spacy download en_core_web_lg

import spacy
import en_core_web_lg
from newsapi import NewsApiClient
import pickle
import pandas as pd
import numpy as np
from collections import Counter 
import nltk
from nltk.tokenize import RegexpTokenizer
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from wordcloud import WordCloud 
import string
#nltk.download('stopwords')
#nltk.download('brown')
#nltk.download('punkt')

nlp_eng = en_core_web_lg.load()
newsapi = NewsApiClient(api_key='92a9a24b2baf4cbfb4b335e29567f375')

temp = newsapi.get_everything(q='coronavirus', language='en', from_param='2022-02-28', to='2020-03-20', sort_by='relevancy')

filename = 'covidArticles.pckl'
pickle.dump(temp, open(filename, 'wb'))
filename = 'covidArticles.pckl'
loaded_model = pickle.load(open(filename, 'rb'))
filepath = '/content/covidArticles.pckl'
pickle.dump(loaded_model, open(filepath, 'wb'))

df = pd.DataFrame(temp['articles'])

tokenizer = RegexpTokenizer(r'\w+')

def getKeywords(token):
  result = []
  punctuation = string.punctuation
  stop_words = stopwords.words('english')
  
  for i in token:
    if (i in stop_words):
      continue
    else:
      result.append(i)
  print(result)
  return result

with open('covidArticles.pckl', 'rb') as f:
    data = pickle.load(f)

print(data)

results = []
for content in df.content.values:
    content = tokenizer.tokenize(content)
    results.append([x[0] for x in Counter(getKeywords(content)).most_common(5)])
df['keywords'] = results

text = str(results)
wordcloud = WordCloud(max_font_size=50, max_words=100, background_color="white").generate(text)
plt.figure()
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.show()

'''Dataset
['Students', 'already', 'going', 'mental', 'health', 'crisis', 'coronavirus', 'turned', 'world', 'upside', 'But', 'pandemic', 'forced', 'everyone', 'isolate', 'exacerbated', '4728', 'chars']
['It', 'January', 'first', 'known', 'case', 'new', 'coronavirus', 'yet', 'named', 'appeared', 'U', 'S', 'Washington', 'state', 'We', 'published', 'explainer', 'coronavirus', 'whether', 'shoul', '777', 'chars']
['Mouthwash', 'designed', 'kill', 'germs', 'mouth', 'It', 'Id', 'say', 'pretty', 'good', 'job', 'And', 'yet', 'seriously', 'trust', 'mouthwash', 'prevent', 'transmission', 'common', 'colds', 'strep', 'infec', '4209', 'chars']
['Navigating', 'air', 'travel', 'COVID', '19', 'pandemic', 'tricky', 'First', 'foremost', 'want', 'make', 'sure', 'youre', 'safe', 'possible', 'On', 'top', 'fewer', 'routes', 'flights', 'availa', '3231', 'chars']
['Reddit', 'pushing', 'back', 'calls', 'company', 'better', 'job', 'moderating', 'disinformation', 'In', 'thread', 'titled', 'Debate', 'dissent', 'protest', 'Reddit', 'CEO', 'Steve', 'Huffman', 'said', 'company', 'woul', '2086', 'chars']
['Earlier', 'summer', 'I', 'began', 'planning', 'familys', 'first', 'real', 'vacation', 'two', 'years', 'I', 'carefully', 'chose', 'National', 'Parks', 'wed', 'visit', 'White', 'Sands', 'Arches', 'Bryce', 'Canyon', 'Capitol', 'Reef', 'Joshua', 'Tree', '4482', 'chars']
['Google', 'require', 'employees', 'get', 'vaccinated', 'coronavirus', 'theyre', 'allowed', 'return', 'companys', 'offices', 'Anyone', 'coming', 'work', 'campuses', 'need', 'vaccinated', '1465', 'chars']
['The', 'way', 'predictions', 'raced', 'ahead', 'experiments', 'Omicrons', 'spike', 'protein', 'reflects', 'recent', 'sea', 'change', 'molecular', 'biology', 'brought', 'AI', 'The', 'first', 'software', 'capable', 'accurately', 'predicting', 'pro', '5116', 'chars']
['When', 'novelcoronavirus', 'first', 'discovered', 'China', 'last', 'winter', 'country', 'responded', 'aggressively', 'placing', 'tens', 'millions', 'people', 'strict', 'lockdown', 'As', 'Covid', '19', 'spread', 'Wuhan', '3647', 'chars']
['Some', 'states', 'lag', 'far', 'behind', 'The', 'Ohio', 'Department', 'Health', 'example', 'notes', 'start', 'finish', 'process', 'collecting', 'sample', 'testing', 'sequencing', 'reporting', 'take', '1943', 'chars']
['Cities', 'across', 'Germany', 'braced', 'major', 'protests', 'coronavirus', 'restrictions', 'Monday', 'tough', 'new', 'vaccine', 'requirement', 'came', 'force', 'Italy', 'governments', 'across', 'Europe', 'continued', 'tigh', '3826', 'chars']
['Researchers', 'trace', 'new', 'variants', 'Africa', 'cases', 'surge', 'India', 'US', 'vaccine', 'rollout', 'progresses', 'even', 'snags', 'Heres', 'know', 'Want', 'receive', 'weekly', 'roundup', 'coronavi', '4174', 'chars']
['Why', 'havent', 'I', 'seen', 'Wayne', 'Knight', 'ads', 'Space', 'Jam', '2', 'Im', 'sorry', 'Mr', 'Knight', 'I', 'cant', 'answer', 'Recently', 'Gov', 'Greg', 'Abbott', 'Republican', 'Texas', 'ended', 'states', 'mask', 'mandate', 'said', '1110', 'chars']
['In', 'China', 'largely', 'contained', 'coronavirus', 'outbreak', 'made', 'big', 'strides', 'returning', 'normal', 'life', 'many', 'people', 'dont', 'feel', 'urgency', 'line', 'vaccine', 'Others', 'wary', 'Ch', '1275', 'chars']
['What', 'think', 'next', 'months', 'look', 'like', 'U', 'S', 'What', 'I', 'think', 'going', 'happen', 'two', 'months', 'vaccinations', 'going', 'everywhere', 'The', 'place', 'going', 'flooded', 'w', '2846', 'chars']
['Similar', 'laborious', 'efforts', 'may', 'scaled', 'country', 'strives', 'herd', 'immunity', 'tries', 'get', 'economy', 'back', 'track', 'seeks', 'return', 'normal', 'way', 'life', 'Vaccines', 'vs', 'tryp', '1331', 'chars']
['But', 'medical', 'ethicists', 'say', 'list', 'misleading', 'suggests', 'risks', 'medical', 'conditions', 'evaluated', 'ranked', 'Is', '50', 'year', 'old', 'Type', '1', 'diabetes', 'higher', 'risk', 'tha', '1300', 'chars']
['How', 'nursing', 'home', 'approach', 'vaccine', 'skeptics', 'At', 'first', 'Tina', 'Sandri', 'chief', 'executive', 'nursing', 'home', 'really', 'leaned', 'heavily', 'Dr', 'Anthony', 'Fauci', 'President', 'Bidens', 'chief', 'medical', 'adv', '1546', 'chars']
['Its', 'scary', 'idea', 'presence', 'variant', 'could', 'make', 'difficult', 'tame', 'Indias', 'disaster', 'A', 'number', 'doctors', 'also', 'saying', 'younger', 'people', 'people', 'fully', 'vaccinate', '2177', 'chars']
['For', 'help', 'I', 'also', 'turned', 'readers', 'asked', 'whats', 'bringing', 'joy', 'days', 'Hundreds', 'wrote', 'Stephen', 'Martin', 'Cave', 'Creek', 'Ariz', 'learned', 'piano', 'age', '76', 'Carol', 'Babc', '2625', 'chars']
'''
