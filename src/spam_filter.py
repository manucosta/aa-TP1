# -*- coding: utf-8 -*- 
import json
import numpy as np
import pandas as pd
import re
from sklearn.tree import DecisionTreeClassifier
from sklearn.cross_validation import cross_val_score
from sklearn.feature_extraction.text import CountVectorizer

import sys

###Consejo: correr esto en una máquina con >4GB de memoria

# Leo los mails (poner los paths correctos).
ham_txt= json.load(open('../dataset_json/ham_dev.json'))
spam_txt= json.load(open('../dataset_json/spam_dev.json'))

# Pongo todos los mails en minusculas
ham_txt = map(lambda x: x.lower(), ham_txt)
spam_txt = map(lambda x: x.lower(), spam_txt)

# Armo el data frame
df = pd.DataFrame(ham_txt+spam_txt, columns=['text'])
df['class'] = ['ham' for _ in range(len(ham_txt))]+['spam' for _ in range(len(spam_txt))]

ham_txt = 0
spam_txt = 0

print "Lei json y arme data frame"

# Extraigo atributos simples
# Agrego feature que clasifica los mails segun tienen o no html
def hasHTML(txt): return "</html>" in txt
df['hasHTML'] = map(hasHTML, df.text)

#) Agrego feature que clasifica los mails segun tienen o no subject
def hasSubject(txt): return "subject:" in txt
df['hasSubj'] = map(hasSubject, df.text)

# Longitud del mail.
df['len'] = map(len, df.text)

# Cantidad de espacios en el mail.
def count_spaces(txt): return txt.count(" ")
df['count_spaces'] = map(count_spaces, df.text)

def extract_body(text):
  text_lines = text.split('\n')
  for line_index in range(len(text_lines)):
      line = text_lines[line_index]
      if line == '\r' or line == '':
        return " ".join(text_lines[line_index+1:])

print "Clasifique por atributos simples"

#Cargo el vocabulario
vocab_file = open("../vocab.txt")
vocab = []

for line in vocab_file:
  vocab.append(line.strip("\n"))

#Armo la matriz de ocurrencias (notar que es una representación para
#matriz esparsa, asi que ocupa relativamente poco)
vectorizer = CountVectorizer(token_pattern=r'[a-z]+', vocabulary=vocab, lowercase=False)
print "Prepare vectorizer"
df_matrix = vectorizer.transform(df.text)
print "Arme matriz"

tags = map(lambda x: "tag-" + x, vectorizer.get_feature_names())
for i in xrange(len(tags)):
  df[tags[i]] = df_matrix[i]
print "Etiquete"

tags.append("hasHTML")
tags.append("hasSubj")
tags.append("len")
tags.append("count_spaces")

print "Empiezo entrenamiento"
# Elijo mi clasificador.
clf = DecisionTreeClassifier(criterion='entropy', max_depth=300)

# Ejecuto el clasificador entrenando con un esquema de cross validation
# de 10 folds.
res = cross_val_score(clf, df_matrix, df['class'], cv=10, n_jobs=2)
print np.mean(res), np.std(res)

# Actualmente da algo como
# 0.992347222222 0.00354504196461