# -*- coding: utf-8 -*- 
import json
import numpy as np
import pandas as pd
import re
from collections import Counter
from sklearn.tree import DecisionTreeClassifier
from sklearn.cross_validation import cross_val_score

import sys

###Consejo: correr esto en una máquina con >4GB de memoria

# Leo los mails (poner los paths correctos).
ham_txt= json.load(open('../dataset_json/ham_dev.json'))
spam_txt= json.load(open('../dataset_json/spam_dev.json'))

df = pd.DataFrame(ham_txt+spam_txt, columns=['text'])
df['class'] = ['ham' for _ in range(len(ham_txt))]+['spam' for _ in range(len(spam_txt))]

def hasHTML(txt): return "</html>" in txt

df['html'] = map(hasHTML, df.text)

print "Lei json"

def ranking(mails, k):
  #Limpio etiquetas html
  mails = [re.sub('<[^<]+?>', '', mail) for mail in mails]
  #Hago una lista con todas las palabras que aparecen (con repeticiones)
  #sin signos de puntuacion ni chirimbolos raros 
  mailWords = []
  for mail in mails:
    for w in re.findall(r'[a-z]+', mail):
      mailWords.append(w)

  countWords = Counter(mailWords)

  return countWords.most_common(k)

def setDifference(spam, ham):
  diff = []
  for w1 in spam:
    counter = 0
    for w2 in ham:
      if w1[0] == w2[0]:
        break
      counter += 1
    if counter == len(ham):
      diff.append(w1)
    counter = 0
  return diff

def countGreaterThan(text, word, threshold):
  return text.count(word) >=  threshold

spam_words = ranking(spam_txt, 400)
ham_words = ranking(ham_txt, 400)
words = []
for word, occur in setDifference(spam_words, ham_words):
  if word == 'html': continue
  df[word] = map(lambda x: countGreaterThan(x, word, 1), df.text)
  words.append(word)


words.append('html')
# Preparo data para clasificar
X = df[words].values
y = df['class']

# Elijo mi clasificador.
clf = DecisionTreeClassifier(criterion='entropy')

# Ejecuto el clasificador entrenando con un esquema de cross validation
# de 10 folds.
res = cross_val_score(clf, X, y, cv=10)
print np.mean(res), np.std(res)
# Actualmente da algo como
# 0.978766666667 0.0161049375145
