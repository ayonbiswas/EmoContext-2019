from __future__ import unicode_literals, print_function, division
from io import open
import unicodedata
import string
import re
import random
import time
import math
import numpy as np
import os,glob
import h5py

from sklearn.model_selection import train_test_split
import torch
from torch.autograd import Variable
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from sklearn.metrics import accuracy_score

from matplotlib import use
use('Agg')
import matplotlib.pyplot as plt

import pandas as pd
import scipy as sp
import numpy as np
# import networkx as nx
from collections import Counter
import operator
import matplotlib.pyplot as plt

import scipy.stats as stats
import math
import itertools
import numbers
import copy

#from IPython.display import display
import matplotlib.pyplot as plt
import pylab as pl
import re

#from IPython.utils import io

import random
random.seed(33)
np.random.seed(33)

import re
import nltk
from nltk.corpus import PortugueseCategorizedPlaintextCorpusReader
from nltk.corpus import PlaintextCorpusReader

import gensim
from gensim.models import LsiModel
from gensim.corpora import Dictionary
from gensim import corpora
from gensim.models import CoherenceModel
from gensim.models.doc2vec import Doc2Vec, TaggedDocument

import sklearn
from sklearn import model_selection, preprocessing, linear_model, naive_bayes, metrics, svm
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn import decomposition, ensemble

import emoji

import pandas as pd

data = pd.read_csv('train.txt', delimiter="\t", encoding='utf-8')

def remove_emoji(text):
    emoji_pattern = re.compile("["
            u"\U0001F600-\U0001F64F"  # emoticons
            u"\U0001F300-\U0001F5FF"  # symbols & pictographs
            u"\U0001F680-\U0001F6FF"  # transport & map symbols
            u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                               "]+", flags=re.UNICODE)

    return(emoji_pattern.sub(r'', text))


# In[38]:


def remove_punct(my_str):
    
    my_str = emoji.demojize(my_str)
    
    punctuations = '''!()-[]{};:'",<>./?@#$%^&*_~'''

    no_punct = ""
    for char in my_str:
        if char not in punctuations:
            no_punct = no_punct + char
    
#     no_punct = remove_emoji(no_punct)

    return no_punct


# In[39]:


class_dict = {'happy':0, 'angry':1, 'sad':2, 'others':3}

y_onehot = np.zeros((4,4),dtype =np.int32)

for i in range(4):
    y_onehot[i,i] = 1


# In[40]:


def one_hot_encoding(string):
    num = class_dict[string]
    return y_onehot[:,num]


# In[41]:


all_text = []
classes = []

for index,row in data.iterrows():
    all_text.append(remove_punct(row['turn1']))
    all_text.append(remove_punct(row['turn2']))
    all_text.append(remove_punct(row['turn3']))
#     classes.append(one_hot_encoding(row['label']))
    classes.append(class_dict[row['label']])

# In[31]:


classes = np.asarray(classes)


documents = [TaggedDocument(doc, [i]) for i, doc in enumerate(all_text)]
model = Doc2Vec(documents,vector_size = 64)



# In[10]:
train = []
for i in range(len(classes)):
    a = np.array([model[3*i],model[3*i+1],model[3*i+2]])
    train.append(a)
    
trainX = np.asarray(train)
np.save("train.npy",trainX)
np.save("label.npy",classes)
#Take 3 continuous sentences from all_text for one observation. The result of the observation is one hot encoded
# trainX = np.load("avg_train.npy")
# classes = np.load("label.npy")



data = pd.read_csv('train.txt', delimiter="\t", encoding='utf-8')



data.turn1[15]



emoji.demojize(data.turn1[15])




def remove_punct(my_str):
    
    my_str = my_str.lower()
    my_str = emoji.demojize(my_str)
    my_str = re.sub(':', ' ', my_str)
    my_str = re.sub('_', ' ', my_str)
    
    punctuations = ''';:'",<>./?@#$%^&*_~'''

    no_punct = ""
    for char in my_str:
        if char not in punctuations:
            no_punct = no_punct + char
    
#     no_punct = remove_emoji(no_punct)

    return no_punct


# In[8]:


class_dict = {'happy':0, 'angry':1, 'sad':2, 'others':3}

y_onehot = np.zeros((4,4))

for i in range(4):
    y_onehot[i,i] = 1


# In[9]:


def one_hot_encoding(string):
    num = class_dict[string]
    return y_onehot[:,num]


# In[10]:


all_text = []
classes = []

for index,row in data.iterrows():
    all_text.append(remove_punct(row['turn1']))
    all_text.append(remove_punct(row['turn2']))
    all_text.append(remove_punct(row['turn3']))
    classes.append(one_hot_encoding(row['label']))


# In[11]:


classes = np.asarray(classes)

from gensim.test.utils import common_texts, get_tmpfile


from nltk import word_tokenize


from gensim.models import Word2Vec


# In[20]:


words_sent = []
for i in all_text:
    tokens = word_tokenize(i)
    words_sent.append(tokens)



# documents = [TaggedDocument(doc, [i]) for i, doc in enumerate(all_text)]
model = Word2Vec(words_sent, size=100, window=5, min_count=1, workers=4)


# # In[23]:


embeddings = []
for i in words_sent:
    tmp = []
    for j in i:
        tmp.append(model[j])
    embeddings.append(tmp)





avg_embeddings = []
for i in words_sent:
    tmp = np.zeros((1,100))
    for j in i:
        tmp += model[j]/len(i)
    avg_embeddings.append(tmp)



trainX = []
for i in range(len(classes)):
    a = np.array([avg_embeddings[3*i][0],avg_embeddings[3*i+1][0],avg_embeddings[3*i+2][0]])
    train.append(a)


# np.save("avg_train",train)


# In[54]:

#trainX = np.load('avg_glove.npy')
print(trainX.shape)
classes = np.argmax(classes,axis = 1)

X_train, X_test, y_train, y_test = train_test_split(trainX, classes, test_size=0.2, random_state=33,stratify=classes)
#classes = np.load("label.npy")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_default_tensor_type('torch.cuda.FloatTensor')
seq_len=3
#look_back=9
batch_size=3016


time_steps = 3

def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

def timeSince(since, percent):
    now = time.time()
    s = now - since
    #print(percent)
    es = s / float(percent)
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))



class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size,num_layers):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers        
        self.LSTM = nn.LSTM(input_size=input_size, hidden_size=hidden_size,num_layers = num_layers,batch_first=True, bidirectional=False)
        self.out1 = nn.Linear(64, 32)
        self.out2 = nn.Linear(32, 4)
        self.soft =  nn.Softmax()
        self.drop = nn.Dropout(p=0.3)
    def forward(self, input, hidden):
        _, (output,cell) = self.LSTM(input, hidden)
        #print(output.size())
        output = output.contiguous().view(3016,-1)
        #print(output.size())
        output = self.out1(output)
        output = self.drop(output)
        output = self.out2(output)
        output = self.drop(output)
        output = self.soft(output)
        return output, hidden

    def initHidden(self,batch_size):
        
        return torch.zeros(num_layers,batch_size, self.hidden_size, device=device)



def train_(input_tensor, target, encoder, encoder_optimizer,batch_size, seq_len):
    #global time_steps
    #print(input_tensor.size())
    h0 = encoder.initHidden(batch_size)
    c0 = encoder.initHidden(batch_size)
   # print(h0.size())
    encoder_optimizer.zero_grad()
    

    output, (hn, cn) = encoder(input_tensor,(h0,c0))
    loss = 0
    #print(output.size())
    values, predicted = torch.max(output, 1,keepdim=True)#.cpu()
    predicted = predicted.view(-1)
    #_, labels = torch.max(target, 0)#.cpu()
    labels = target
    #correct += (predicted == labels).sum().item()
    #print(labels.size())
    #print(predicted.size())
    
    loss = loss_(output,Variable(labels.long()))
    acc = accuracy_score(labels,predicted ) 
    loss.backward()

    encoder_optimizer.step()
    

    return loss.item(),acc

def test_(input_tensor, target, encoder,batch_size):
    #global time_steps
    #print(input_tensor.size())
    h0 = encoder.initHidden(batch_size)
    c0 = encoder.initHidden(batch_size)
   # print(h0.size())
    #encoder_optimizer.zero_grad()
    

    output, (hn, cn) = encoder(input_tensor,(h0,c0))
    loss = 0
    #print(output.size())
    values, predicted = torch.max(output, 1,keepdim=True)#.cpu()
    predicted = predicted.view(-1)
    #_, labels = torch.max(target, 0)#.cpu()
    labels = target
    #correct += (predicted == labels).sum().item()
    #print(labels.size())
    #print(predicted.size())
    
    loss = loss_(output,Variable(labels.long()))
    acc = accuracy_score(labels,predicted ) 
    #loss.backward()

    #encoder_optimizer.step()
    

    return loss.item(),acc


def trainIters(encoder,train,test,X_test,y_test, epoch,n_iters, batch_size,print_every=25, plot_every=25, learning_rate=0.001):
    global seq_len
    start = time.time()
    plot_losses = []
    print_loss_total = 0  # Reset every print_every
    plot_loss_total = 0  # Reset every plot_every
    accuracy = 0
    #encoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate)
    encoder_optimizer = optim.Adam(encoder.parameters(), lr = 0.001)
    
    for e in range(epoch):

        for iter in range( 1,n_iters+1 ):
            #rand = np.random.choice(train.shape[0], batch_size, replace=False)
            #print(rand)
            ctr = iter -1
            train_samples = train[ctr*batch_size:(ctr+1)*batch_size, :,:]
            test_samples = test[ctr*batch_size:(ctr+1)*batch_size]
            randperm = torch.randperm(batch_size)
            train_samples =  train_samples[randperm]
            test_samples =  test_samples[randperm]
            #training_pair = training_pairs[iter - 1]
            input_tensor = Variable(torch.Tensor(train_samples), requires_grad=True)
            target_tensor = Variable(torch.Tensor(test_samples), requires_grad=False )
    #         print(input_tensor.size())
    #         print(target_tensor.size())
            loss ,acc= train_(input_tensor, target_tensor, encoder, encoder_optimizer,batch_size,seq_len)
            print_loss_total += loss
            plot_loss_total += loss
            if(iter%print_every==0):
                print("\n")
            else:
                print(".",end=" ")
            if iter % print_every == 0:
                print_loss_avg = print_loss_total #/ print_every
                print_loss_total = 0
                
                print('%s (%d %d%%) loss :%.8f epoch : %d' % (timeSince(start, float(iter) / n_iters),
                                             iter, iter / n_iters * 100, print_loss_avg,e))
                #print(accuracy)

            if iter % plot_every == 0:
                plot_loss_avg = plot_loss_total / plot_every
                plot_losses.append(plot_loss_avg)
                plot_loss_total = 0
            accuracy += acc
        accuracy = accuracy/n_iters
        
        print("\ntrain accuracy : ",accuracy)
        test_tot_acc = 0
        for i in range(0,2):
            X_test_samples = X_test[i*batch_size:(i+1)*batch_size, :,:]
            y_test_samples = y_test[i*batch_size:(i+1)*batch_size]
            input_tensor_test = Variable(torch.Tensor(X_test_samples), requires_grad=True)
            target_tensor_test = Variable(torch.Tensor(y_test_samples ), requires_grad=False )
            loss_test ,acc_test= test_(input_tensor_test, target_tensor_test, encoder,batch_size)
            test_tot_acc += acc_test
        print("test accuracy : ",test_tot_acc/2.)
        
        


    


loss_ = nn.CrossEntropyLoss()
hidden_size = 64
num_layers = 1
encoder = EncoderRNN(50, hidden_size,num_layers).to(device)


trainIters(encoder,X_train,y_train,X_test,y_test, batch_size = 3016,epoch  = 500 ,n_iters = 8, print_every=5)


out_path_en = "model_enc.pth"

torch.save(encoder.state_dict, out_path_en)

#evaluate(testX,testY,fname,encoder1,batch_size = 4200, n_iters = 30, seq_len=32)




