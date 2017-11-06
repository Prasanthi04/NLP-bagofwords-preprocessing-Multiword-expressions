# -*- coding: utf-8 -*-
"""
Created on Thu Sep 21 03:16:45 2017

@author: NAGA
"""
import numpy as np
import pandas as pd
import urllib.request as urllib2 
from nltk import word_tokenize, sent_tokenize,pos_tag, TweetTokenizer, MWETokenizer
from nltk import punkt, bigrams, collocations
from bs4 import BeautifulSoup 
from nltk.corpus import stopwords
import re
import collections 
from collections import Counter
from textblob import TextBlob

url = "http://www.site.uottawa.ca/~diana/csi5386/A1_2017/microblog2011.txt"
response = urllib2.urlopen(url)
raw = response.read().decode('utf8')
t=TweetTokenizer()
rawstr = str(raw)
tok = t.tokenize(rawstr.lower())

print("The total number of raw tokens is :", len(tok))

print("The total number of unique tokens found:", len(Counter(tok)))

print("The ratio of unique tokens and total tokens:", len(Counter(tok))/len(tok))

#WRITING COUNTER TO OUTPUT FILE
import pprint
with open('Tokens.txt', 'w', errors ='ignore') as fout:
   fout.write(pprint.pformat(Counter(tok)))

###Removing punctuations
tokens_words = re.compile("[A-Za-z0-9]")
tokens_words_alone = filter(tokens_words.match,tok)
tokens_words_list = list(tokens_words_alone)
print(tokens_words_list)

words_alone_count = Counter(tokens_words_list)
print("Total word count after removing punctuations and special characteristics is:\n", len(words_alone_count))
print("The ratio of unique word only tokens to total number of word_only tokens is:\n", len(words_alone_count)/len(tokens_words_list))


stop = list(stopwords.words('english'))
words_alone_nostop = [i for i in tokens_words_list if i not in stop]
print(words_alone_nostop)
words_alone_nostop_count = Counter(words_alone_nostop)
print(words_alone_nostop_count)
myCount_nostop = pd.Series(words_alone_nostop).value_counts()
print(myCount_nostop)

######################################################################3
#token formatting for results purporse
####################################################################
from itertools import chain
import io
tknzr = TweetTokenizer()
s1 = io.open('microblog2011.txt', 'r', errors='ignore', encoding='utf8')
s2 = s1.readline()
t3 = []
for s2 in s1:
    t=tknzr.tokenize(str(s2))
    t3.append(t)
 
for sublist in t3:
    print(" ".join(val for val in sublist if not val.isspace()))
  
##############################################################333
#Method for finding pair of words
###############################################################
    
def remove_punc_stop(token_func):
    
    words_al = re.compile("[A-Za-z0-9]")
    words_alone = filter(words_al.match,token_func)
    words_alone_list = list(words_alone)
    stop = list(stopwords.words('english'))
    words_alone_nostop = [i for i in words_alone_list if i not in stop]
    #print(words_alone_nostop)
    words_p = re.findall('\w+', str(words_alone_nostop))
    pairs = bigrams(words_p) 
    pairs_list = list(pairs)
    print(pairs_list)
    return pairs_list

  
text = open('microblog2011.txt', mode='r', errors = 'ignore')

tken = []
pci=[]
t = text.readline()
for t in text:
    tknzr = TweetTokenizer()
    tknzr.tokenize(str(t))
    tkn=tknzr.tokenize(str(t))
    tken.append(tkn)
    #print(tkn)
    tknl = [x.lower() for x in tkn]
    pc = remove_punc_stop(tknl)
    pci.append(pc)
#print(pci)

pair_c = Counter(x for xs in pci for x in set(xs))
print("The total number of pairs with frequencies", pair_c)

print(pair_c.most_common(100)) 

total_pci_count = 578494
total_pairc_count = len(pair_c)

lexical_density = total_pairc_count/total_pci_count
print("The lexical density is ", lexical_density)


##############################################################################333
##MWE EXPRESSIONS#####3
###################################################################################

from gensim.models import Phrases
from gensim.models.phrases import Phraser
from nltk.corpus import stopwords

def remove_punc_stop_mwe(token_func):
    
    words_al = re.compile("[A-Za-z0-9]")
    words_alone = filter(words_al.match,token_func)
    words_alone_list = list(words_alone)
    stop = list(stopwords.words('english'))
    words_alone_nostop = [i for i in words_alone_list if i not in stop]
    return words_alone_nostop

text1 = open('microblog2011.txt', mode='r', errors = 'ignore')
t2 = text1.readline()

t1 = []

for t2 in text1:
    tknzr1 = TweetTokenizer()
    tken1 = tknzr1.tokenize(str(t2))
    print(tken1)
    tkenl = [x.lower() for x in tken1]
    token_process_mwe = remove_punc_stop_mwe(tkenl)
    t1.append(token_process_mwe)
    
bigram = Phrases(t1, min_count=5, threshold=3, delimiter=b' ')
bigram_phraser = Phraser(bigram)

tokens_mwe = []
for sent in t1:
    tokens_ = bigram_phraser[sent]
    print(tokens_)
    tokens_mwe.append(tokens_)
    
print(tokens_mwe)


c = Counter(x for xs in tokens_mwe for x in set(xs))

import pprint
with open('mwe_tokens.txt', 'w', errors='ignore') as f:
    f.write(pprint.pformat(c))
    
