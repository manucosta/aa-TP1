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
ham_txt= json.load(open('../dataset_dev/ham_dev.json'))
spam_txt= json.load(open('../dataset_dev/spam_dev.json'))

df = pd.DataFrame(ham_txt+spam_txt, columns=['text'])
df['class'] = ['ham' for _ in range(len(ham_txt))]+['spam' for _ in range(len(spam_txt))]

def hasHTML(txt): return "</html>" in txt

df['html'] = map(hasHTML, df.text)

print "Lei json"

def ranking(mails, k):
  #Limpio etiquetas html
  # l = [re.sub('<[^<]+?>', '', mail) for mail in mails]

  l = []
  for mail in mails:
    mail = " ".join(mail)
    l.append(re.sub('<.*>', '', mail))

  #Hago una lista con todas las palabras que aparecen (con repeticiones)
  #sin signos de puntuacion ni chirimbolos raros 
  mailWords = []
  for mail in l:
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

def filtrarMugresGenerico(mailList, bodys, subjects, froms, tos):
  for mail in mailList:
    mail_lines = mail.split('\n')
    for line_index in range(len(mail_lines)):
      line = mail_lines[line_index]
      if line[:5] == "from:":
        froms.append(line[6:])
      elif line[:8] == "subject:":
        subjects.append(line[9:])
      elif line[:3] == "to:":
        tos.append(line[4:])
      elif line == "\r":
        bodys.append(mail_lines[line_index:])
        break

# def frecuenciasPromedio(spamList, words):

body_spam = []
subjetc_spam = []
from_spam = []
to_spam = []
filtrarMugresGenerico(spam_txt, body_spam, subjetc_spam, from_spam, to_spam)

body_ham = []
subjetc_ham = []
from_ham = []
to_ham = []
filtrarMugresGenerico(ham_txt, body_ham, subjetc_ham, from_ham, to_ham)

body_spam_words = ranking(body_spam, 400)

body_ham_words = ranking(body_ham, 400)

words = []
for word, occur in setDifference(body_spam_words, body_ham_words):
  print word, occur
  if word == 'html': continue
  df[word] = []
  for mail in df.text:
    if word == 'base':
      print mail
    df[word].append(countGreaterThan(mail, word, 1)) 
  # df[word] = map(lambda x: countGreaterThan(x, word, 1), df.text)
  words.append(word)


words.append('html')
# Preparo data para clasificar
X = df[words].values
y = df['class']


sys.exit()

# Elijo mi clasificador.
clf = DecisionTreeClassifier(criterion='entropy')

# Ejecuto el clasificador entrenando con un esquema de cross validation
# de 10 folds.
res = cross_val_score(clf, X, y, cv=10)
print np.mean(res), np.std(res)
# Actualmente da algo como
# 0.978766666667 0.0161049375145