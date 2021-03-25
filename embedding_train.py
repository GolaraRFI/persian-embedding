###############################  Import libraries #######################################
###### array modules ######
import pandas as pd
import numpy as np
import re

###### visualization moduls #####
from sklearn.decomposition import PCA
from matplotlib import pyplot
from tabulate import tabulate
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from mpl_toolkits.mplot3d import Axes3D

###### basic modules ######
from collections import Counter
import tqdm, os, gc, subprocess, zipfile, gdown, pickle
#!pip install gdown
from pathlib import Path
from termcolor import colored

###### tensorflow modules ######
import tensorflow as tf
from tensorflow import keras

## Embedding Visualization Modules 
from tensorboard.plugins import projector
import tensorboard

###### embedding creation modules ######
import gensim
import gensim.models.keyedvectors as word2vec
from gensim.models import FastText
from gensim.models import Word2Vec

###### Text Processing Tools Modules ######
import spacy
from hazm import *
# in the future you can import step_1

## modules initialization ##
stemmer = Stemmer()
lemmatizer = Lemmatizer()
sent_tokenizer = SentenceTokenizer()
normalizer = Normalizer()
tokenizer = WordTokenizer()
stop_w_list = stopwords_list()

#################################### Functions ##########################################

def un_zip(path):
    if not os.path.exists(path):
      with zipfile.ZipFile(path+'.zip','r') as zip_ref:
          zip_ref.extractall(path)

def preprocessing_data(line, remove_stop_word=False, lemmatize=False, stem=False):
    line = normalizer.normalize(line)
    line = tokenizer.tokenize(line)
    if remove_stop_word:
        line = [word for word in line if word not in stop_w_list]
    if stem:
        line = [stemmer.stem(word) for word in line]
    if lemmatize:
        line = [lemmatizer.lemmatize(word) for word in line]
        line
    line = ' '.join(line)
    return line

def create_embeddings(sentences,embedding_path='defualt_path.gensimmodel', word_vec_type='fasttext', **params):
    if word_vec_type == 'fasttext':
        model = FastText(sentences, **params)
    else:
        model = Word2Vec(sentences, **params)
    print('Saving the embedding models')
    model.wv.save_word2vec_format(embedding_path)
    return model

def fasttext(sentence_text,embedding_path,embedding_dim,MYSG,iteration,mincount):
   return create_embeddings(sentence_text,embedding_path=embedding_path + '.gensimmodel' ,
                            size=embedding_dim, word_vec_type='fasttext',
                            sg=MYSG, workers=1,
                            iter=iteration,min_count=mincount,
                            word_ngrams=1, min_n=2, max_n=8, bucket=2000000,)

def w2v(sentence_text,embedding_path,embedding_dim,MYSG,iteration,mincount,window_size):
        return create_embeddings(sentence_text,embedding_path=embedding_path + '.gensimmodel',
                                  size=embedding_dim, word_vec_type='w2v',
                                  sg=MYSG, workers=1,min_count=mincount,window=window_size,
                                  iter=iteration,)



################################## Train models #########################################

# un_zip("Persian_Corpus.zip")

data_path = "Persian_Corpus.txt"

f = open(data_path, "r")
data = []
for line in f:
  data.append(line)
f.close()

print("Data size: ",len(data),"\n",data[:10])

# cleaning data
cleaned_data = []
for doc in tqdm.tqdm(data):
  cleaned_data.append(preprocessing_data(doc, remove_stop_word=True,lemmatize=False,stem=False))

print("Uncleaned data: ",data[10])
print("Cleaned data: ",cleaned_data[10])

# spliting data
splited_line = [line.split(sep=' ') for line in cleaned_data if len(line.split(sep=' ')) > 3]
print(splited_line[9])

#################################### Train and save models ##############################

# train a word2vec model 1
embedding_path = 'sample_w2v_4'
w2v_model = w2v(splited_line,embedding_path=embedding_path, embedding_dim = 50, MYSG=0, iteration=2, mincount=20, window_size=5)

# train a word2vec model 2
embedding_path = 'sample_w2v_4'
w2v_model = w2v(splited_line,embedding_path=embedding_path, embedding_dim = 50, MYSG=0, iteration=2, mincount=20, window_size=10)

# train a word2vec model 3
embedding_path = 'sample_w2v_4'
w2v_model = w2v(splited_line,embedding_path=embedding_path, embedding_dim = 100, MYSG=0, iteration=2, mincount=20, window_size=5)

# train a word2vec model 4
embedding_path = 'sample_w2v_4'
w2v_model = w2v(splited_line,embedding_path=embedding_path, embedding_dim = 100, MYSG=0, iteration=2, mincount=20, window_size=10)

# train a fasttext model 5
embedding_path = 'sample_fasttext_4'
fastext_model = fasttext(splited_line, embedding_path=embedding_path, embedding_dim = 50, MYSG=1, iteration=5, mincount=20)

# train a fasttext model 6
embedding_path = 'sample_fasttext_4'
fastext_model = fasttext(splited_line, embedding_path=embedding_path, embedding_dim = 50, MYSG=0, iteration=5, mincount=20)

# train a fasttext model 7
embedding_path = 'sample_fasttext_4'
fastext_model = fasttext(splited_line, embedding_path=embedding_path, embedding_dim = 100, MYSG=1, iteration=5, mincount=20)

# train a fasttext model 8
embedding_path = 'sample_fasttext_4'
fastext_model = fasttext(splited_line, embedding_path=embedding_path, embedding_dim = 100, MYSG=0, iteration=5, mincount=20)

