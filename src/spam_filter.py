# -*- coding: utf-8 -*- 
import json
import numpy as np
import pandas as pd
import re
from sklearn.tree import DecisionTreeClassifier
from sklearn.cross_validation import cross_val_score
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn import ensemble
from sklearn.grid_search import GridSearchCV

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

'''
def strangeChars(txt):
  print "s"
  for c in txt:
    if(c in [u'á',u'é',u'í',u'ó',u'ú',u'ñ',
             u'ä',u'ë',u'ï',u'ö',u'ü'] or
        ord(c) in range(187, 243)):
      return True
  return False
df['str_chars'] = map(strangeChars, df.text)
'''

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

X = df_matrix
y = df['class']

print "Empiezo entrenamiento"
# Defino los clasificadores

# Decision Tree
dectree = DecisionTreeClassifier()
dectree_param_grid = {"max_depth":[None, 3, 10, 100, 300], 
			"max_features": [None, 1, 3, 10],
			"max_leaf_nodes": [None, 25, 50, 100, 1000],
			"min_samples_split": [2, 3, 4, 10],
			"criterion": ["gini", "entropy"]}

# Naive Bayes
gnb = GaussianNB()
gnb_param_grid = {}

# KNN
neigh = KNeighborsClassifier()
neigh_param_grid = {}

# SVM
supvecmac = svm.SVC()
supvecmac_param_grid = {}

# Random Forest
ranfor = ensemble.RandomForestClassifier()
ranfor_param_grid = {}

clasifs = [dectree, gnb, neigh, supvecmac, ranfor]
params_grid = [dectree_param_grid, gnb_param_grid, neigh_param_grid, supvecmac_param_grid, ranfor_param_grid]
names = ["Decision Tree", "Gaussian Naive Bayes", "K Nearest Neighbors", "Support Vector Machines", "Random Forest"]

# Ejecuto el clasificador entrenando con un esquema de cross validation
# de 10 folds.
for (clf, param_grid, name) in zip(clasifs, params_grid, names):
	grid_search = GridSearchCV(clf, param_grid = param_grid, cv = 10)

	# En el caso de Naive Bayes, es necesaria una matriz densa
	if clf == gnb:
		grid_search.fit(X.todense(), y)
	else:
		grid_search.fit(X, y)

	print name
	print grid_search.best_params_
	#for params, mean_score, scores in grid_search.grid_scores_:
	#	print("%0.3f (+/-%0.3f) for %f" % (mean_score, scores.std(), params))
	print grid_search.best_score_
	
	# res = cross_val_score(clf, df_matrix, df['class'], cv=10, n_jobs=1)
	# print np.mean(res), np.std(res)
	# Actualmente da algo como
	# 0.990988888889 0.0061120100349