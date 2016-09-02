import json
import numpy as np
import pandas as pd
import re
from collections import Counter
from sklearn.tree import DecisionTreeClassifier
from sklearn.cross_validation import cross_val_score

import sys

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
  intersec = []
  for w1 in spam:
    for w2 in ham:
      if w1[0] == w2[0]:
        intersec.append(w1)
        break
  return [x for x in spam if x not in intersec]   

def occurrencesOf(text, word, ocurr):
  return text.count(word) >=  (float(ocurr)/len(spam_txt))

spam = ranking(spam_txt, 100)
ham = ranking(ham_txt, 100)
words = []
for word, ocurr in setDifference(spam, ham):
  if word == 'html': continue
  df[word] = map(lambda x: occurrencesOf(x, word, ocurr), df.text)
  words.append(word)

'''
Esto anda bien
spam_words = ranking(spam_txt[0:20], 100)
ham_words = ranking(ham_txt[0:20], 100)
print setDifference(spam_words, ham_words)
sys.exit(0)
'''
# Preparo data para clasificar
X = df[[words.append('html')]].values
y = df['class']

# Elijo mi clasificador.
clf = DecisionTreeClassifier()

# Ejecuto el clasificador entrenando con un esquema de cross validation
# de 10 folds.
res = cross_val_score(clf, X, y, cv=10)
print np.mean(res), np.std(res)
# Clasificando solo por html:
# Dataset viejo: 0.924360236207 0.00416089614975
# Dataset nuevo: 0.752233333333 0.144342044472