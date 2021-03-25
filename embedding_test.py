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

def test_most_similar_analogy(model, in_word_1, in_word_2, out_word_1, top_number=3):
  if in_word_1 not in model.wv.vocab.keys():
    print('\nfirst word is not in vocab')
    return [],[],[],[]
  if in_word_2 not in model.wv.vocab.keys():
    print('\nsecond word is not in vocab')
    return [],[],[],[]
  if out_word_1 not in model.wv.vocab.keys():
    print('\nthird word is not in vocab')
    return [],[],[],[]

  in_1 = model.wv.vectors[model.wv.vocab[in_word_1].index]
  in_1_2 = in_1 / sum(in_1 ** 2) ** 0.5

  in_2 = model.wv.vectors[model.wv.vocab[in_word_2].index]
  in_2_2 = in_2 / sum(in_2 ** 2) ** 0.5

  out_1 = model.wv.vectors[model.wv.vocab[out_word_1].index]
  out_1_2 = out_1 / sum(out_1 ** 2) ** 0.5

  out_2_2 = in_2 + out_1 - in_1
  out_2_3 = in_2_2 + out_1_2 - in_1_2

  distance_by_vec = model.wv.similar_by_vector(out_2_2, topn=top_number)
  normalized_distance_by_vec = model.wv.similar_by_vector(out_2_3, topn=top_number)
  most_similar_distance = model.wv.most_similar(positive=[in_word_2, out_word_1], \
                                    negative=[in_word_1], topn=top_number)
  most_similar_cosmul_distance = model.wv.most_similar_cosmul(positive=[in_word_2, out_word_1],\
                                    negative=[in_word_1], topn=top_number)

  list0 = [related_word[0] for related_word in distance_by_vec]
  list1 = [related_word[0] for related_word in normalized_distance_by_vec]
  list2 = [related_word[0] for related_word in most_similar_distance]
  list3 = [related_word[0] for related_word in most_similar_cosmul_distance]

  print('\ndistance by vector : ', list0)
  print('normaled distance by vector: ', list1)
  print('most similar distance : ', list2)
  print('most similar cosmul distance : ', list3)

  return list0,list1,list2,list3

def test_model(mymodel, mytopnumber):
  right0,right1,right2,right3 = [0]*(len(data)),[0]*(len(data)),[0]*(len(data)),[0]*(len(data))

  for i in range(len(data)):
    in_word_1, in_word_2, out_word_1 = data[i][1],data[i][2],data[i][3]
    list0,list1,list2,list3 = test_most_similar_analogy(mymodel, in_word_1, in_word_2, out_word_1, top_number=mytopnumber)
    
    for j in range(len(list0)):
        if (list0[j] == data[i][4]):
          right0[i] += 1
    for j in range(len(list1)):
        if (list1[j] == data[i][4]):
          right1[i] += 1
    for j in range(len(list2)):
        if (list2[j] == data[i][4]):
          right2[i] += 1
    for j in range(len(list3)):
        if (list3[j] == data[i][4]):
          right3[i] += 1

  list_right = [right0,right1,right2,right3]
  list_performance = [0]*4
  for i in range(4):
    num_nonzero, num_zero = 0, 0
    for ele in list_right[i]: 
        if ele != 0:
          num_nonzero += ele
        else:
          num_zero += 1
    list_performance[i] = (num_nonzero*100)/num_zero
  print("---------------------------End of testing Model----------------------------------")
  return list_performance



###################################### Test models ######################################

f = open("Test_file.txt", "r")
data = []
for line in f:
  wordList = re.sub("[^\w]", " ",  line).split()
  data.append(wordList)
f.close()

# determine top number value
mytopnumber = 20

list_path = ['sample_w2v_1.gensimmodel','sample_w2v_2.gensimmodel','sample_w2v_3.gensimmodel','sample_w2v_4.gensimmodel','sample_fasttext_1.gensimmodel','sample_fasttext_2.gensimmodel','sample_fasttext_3.gensimmodel','sample_fasttext_4.gensimmodel']
ListPerformance = []
for model_num in range(8):
  MyModel = word2vec.KeyedVectors.load_word2vec_format(list_path[model_num], binary=False)
  ListPerformance.append(test_model(MyModel, mytopnumber))

list_model_name = ['word2vec1','word2vec2','word2vec3','word2vec4','FastText1','FastText2','FastText3','FastText4']
listx = ListPerformance
listx = np.array(listx)
listx = listx.transpose()

a = np.array([list_model_name])
b = [listx[0]]
c = [listx[1]]
d = [listx[2]]
e = [listx[3]]

Matrix = np.concatenate((a, b, c, d, e))
Matrix = [[Matrix[j][i] for j in range(len(Matrix))] for i in range(len(Matrix[0]))]
print(tabulate(Matrix, headers=["Model", "distance_by_vec", "normalized_distance_by_vec","most_similar_distance","most_similar_cosmul_distance"], tablefmt="grid"))

list_per = [b,c,d,e]
headers=['distance_by_vec', 'normalized_distance_by_vec','most_similar_distance','most_similar_cosmul_distance']

j = 0
for i in list_per:
  labels = ['W2V1','W2V2','W2V3','W2V4','FT1','FT2','FT3','FT4']
  X = np.array(i)
  X = np.squeeze(X)
  width = 0.35

  fig, ax = plt.subplots()
  plt.ylim([0, 40])
  ax.bar(labels, X, width)
  ax.set_ylabel('Performance')
  ax.set_title(headers[j])
  j += 1

  plt.show()