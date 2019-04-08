import json
import re
import bz2
import regex
from tqdm import tqdm
from scipy import sparse
from collections import Counter
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import *
from pymorphy2 import MorphAnalyzer

import pandas as pd
import numpy as np
import nltk
from nltk.tokenize import sent_tokenize
from gensim.models import Word2Vec
import matplotlib.pyplot as plt
import seaborn as snsА
from gensim.corpora import *
from gensim.models import  *
from gensim import similarities
from pylab import pcolor, show, colorbar, xticks, yticks

from nltk.stem.snowball import RussianStemmer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

from sklearn.feature_extraction.text import CountVectorizer

from sklearn.decomposition import TruncatedSVD

from sklearn.manifold import TSNE
from sklearn.preprocessing import Normalizer
from sklearn.feature_extraction.text import *
import igraph as ig
import seaborn as sns

import warnings
warnings.filterwarnings('ignore')

nltk.download('punkt')
model = None

#параметр ограничения данных для ускорения расчетов
AMOUNT = 2000

def main(df):
    global model, g
 
    texts = df.text.to_list()
    #1.Разбейте всю коллекцию отзывов на предложения. Лемматизируйте все слова.

    # функция для удаления стоп-слов
    mystopwords = stopwords.words('russian') + ['это', 'наш' , 'тыс', 'млн', 'млрд', 'также',
                                                'т', 'д', 'который','прошлый', 'сей', 'свой',
                                                'мочь', 'в', 'я', '-', 'мой', 'ваш', 'и', '5']
    def remove_stopwords(text, mystopwords = mystopwords):
        try:
            return " ".join([token for token in text.lower().split() if not token in mystopwords])
        except:
            return ""

    # функция лемматизации
    def lemmatize(text, morph=MorphAnalyzer()):
        try:
            lemmas = [morph.parse(word)[0].normal_form for word in text.split()]
            return ' '.join(lemmas)
        except:
            return ""
    
    pattern = re.compile('[а-яА-Я]+')
    
    def only_words(text, p=pattern):
        return ' '.join(p.findall(text)).strip()

    #разбиваем каждый текст на предложения и помещаем в один список
    sentences = []
    pattern = re.compile('[а-яА-Я]+')
    for text in texts:
        text = lemmatize(remove_stopwords(text))
        text_sentences = sent_tokenize(text)
        for sentence in text_sentences:
            sentences.append(pattern.findall(sentence))

    model = Word2Vec(min_count=1)
    model.build_vocab(sentences)
    model.train(sentences, total_examples=model.corpus_count, epochs=model.iter)
    #определение ближайших слов
    result1 = model.wv.most_similar(positive="банк", topn=10)
    print('result1 = ', result1)
    #аналогии 
    result2 = model.wv.most_similar(positive=['кредит', 'вклад'], negative=['долг'])
    print('result2 = ', result2)
    #определение лишнего слова
    result3 = model.wv.doesnt_match("банк перевод счет отделение".split())
    print('result3 = ', result3)

    #Полученные результаты (AMOUNT=10000)
    #result1 =  [('клик', 0.655503511428833), ('банком', 0.6381771564483643), ('банка', 0.5996867418289185), ('мобайл', 0.5682080984115601), ('банку', 0.5554714202880859), ('клике', 0.5553926229476929), ('клика', 0.5493252873420715), ('беларусь', 0.545136570930481), ('банке', 0.5433052778244019), ('терроризирует', 0.5427347421646118)]
    #result2 =  [('депозит', 0.7282594442367554), ('посочувствовали', 0.6706955432891846), ('вклада', 0.6341916918754578), ('автокопилку', 0.6184648871421814), ('депозита', 0.6164340972900391), ('вклады', 0.6158702373504639), ('преддефолтный', 0.6100568771362305), ('баррикадной', 0.6076372861862183), ('ргают', 0.6062372922897339), ('потребкредить', 0.5898075103759766)]
    #result3 =  отделение
    
    
    df['text_without_stopwords'] = df.text.apply(remove_stopwords)
    df['lemmas'] = df['text_without_stopwords'].apply(lemmatize)
    df['lemmas'] = df['lemmas'].apply(remove_stopwords)


    vectors = TfidfVectorizer(max_features=500).fit_transform(df['lemmas'][:AMOUNT])
    X_reduced = TruncatedSVD(n_components=5, random_state=40).fit_transform(vectors)
    X_embedded = TSNE(n_components=2, perplexity=5, verbose=0).fit_transform(X_reduced)

    vis_df = pd.DataFrame({'X': X_embedded[:200, 0], 'Y': X_embedded[:200, 1], 'topic' : df.title[:200]})
    
    #визуализация TSNE
    g = sns.FacetGrid(vis_df, hue="topic", size=10).map(plt.scatter, "X", "Y").add_legend()
    g.savefig("tsne.png")

    #красный цвет - проблемы с он-лайн обслуживанием, зеленый цвет - отказ банка

    
    #визуализация банков на плоскости
    v1 = model['хороший'] - model['плохой']
    v2 = model['быстрый'] - model['медленный']
    banks = ['сбербанк', 'втб', 'тинькофф', 'россельхозбанк', 'росбанк', 'авангард', 'ситибанк', 'альфабанк']
    banks_x = []
    banks_y = []
    for bank in banks:
        banks_x.append(np.dot(v1, model[bank]))
        banks_y.append(np.dot(v2, model[bank]))
        
    fig, ax = plt.subplots()
    ax.scatter(banks_x, banks_y)
    
    for i, txt in enumerate(banks):
        ax.annotate(txt, (banks_x[i], banks_y[i]))

    ax.set(xlabel='плохо-хорошо', ylabel='медленно-быстро')
    fig.savefig('plane.png')
        
    #plt.show()

    # пример построения графа

    keys = list(model.wv.vocab.keys())[:AMOUNT]
   
    g = ig.Graph(directed=True)
    labels = []
    fixes = []
    weights = []
    
    positive_words = ['любезно', 'готовый', 'хороший', 'уважаемый', 'положительный', 'выбор']
    negative_words = ['беспокоить', 'достает', 'неважно', 'неграмотность', 'никак', 'просрочить']
    
        
    for word in keys:
        label = -1  #непомеченные слова
        fix = False
        if  word in positive_words:
            label = 1  #положительная метка
            fix = True
        if word in negative_words:
            label = 0  #отрицательная метка
            fix = True
            
        labels.append(label)
        fixes.append(True)
        
        g.add_vertex(word)
    
          
    
    for word in keys:
        node = g.vs.select(name = word).indices[0]
        similar_words = model.most_similar(word, topn=10)
        for sim in similar_words:
            try:
                word1 = sim[0]
                val  = sim[1]
                new_node = g.vs.select(name = word1).indices[0]
                g.add_edge(node, new_node, weight = val)
                weights.append(val)
            
            except Exception as err:
                print('Error', err)

    m = g.community_label_propagation(initial=labels, weights=weights, fixed=fixes)
    print('membership = ', m.membership)  # массив меток слов
    print('labels = ', labels)
    print('weights = ', weights)
    print('len weights = ', len(weights))
    return m


def start():
    #путь к файлу (изменить по необходимости)
    df = pd.read_json(r'../hw16/banki_responses.json', lines=True)
    df = df.iloc[:AMOUNT]
    return main(df)


if __name__ == '__main__':
    m = start()








