#!/usr/bin/python
# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from utilities import *
from sklearn.tree import DecisionTreeClassifier
from sklearn.cross_validation import cross_val_score, train_test_split
from sklearn.naive_bayes import BernoulliNB
from sklearn import ensemble
from sklearn.feature_selection import SelectKBest, chi2

def classifier_barplot(raw_result, kbest_result, pca_result):
  N = 6
  rawMeans = raw_result

  ind = np.arange(N)  # the x locations for the groups
  width = 0.23      # the width of the bars

  fig, ax = plt.subplots()
  rects1 = ax.bar(ind, rawMeans, width, color='r')

  kbestMeans = kbest_result
  rects2 = ax.bar(ind + width, kbestMeans, width, color='y')

  pcaMeans = pca_result
  rects3 = ax.bar(ind + 2*width, pcaMeans, width, color='b')

  # add some text for labels, title and axes ticks
  ax.set_ylabel('Score')
  ax.set_title('Scores por clasificador y dimensionalidad')
  ax.set_xticks(ind + width)
  ax.set_xticklabels(('Decision Tree', 'Gaussian NB', 'Multinomial NB', 'Bernoulli NB', 'KNN', 'Random Forest'), rotation=20)

  ax.legend((rects1[0], rects2[0], rects3[0]), (u'Sin reducción de dimensionalidad', u'Con K-Best, K=100', u'Con PCA, n_components = 142'))

  plt.ylim((0,1.5))

  def autolabel(rects):
      # attach some text labels
      for rect in rects:
          height = rect.get_height()
          ax.text(rect.get_x() + rect.get_width()/2., 1.02*height,
                  '%.4f' % float(height),
                  ha='center', va='bottom', rotation='vertical')

  autolabel(rects1)
  autolabel(rects2)
  autolabel(rects3)

  plt.savefig('barplot.pdf')

def feature_lineplot():
  # Leo los mails (poner los paths correctos).
  ham_txt= json.load(open('../dataset_json/ham_dev.json'))
  spam_txt= json.load(open('../dataset_json/spam_dev.json'))

  # Pongo todos los mails en minusculas
  ham_txt = map(lambda x: x.lower(), ham_txt)
  spam_txt = map(lambda x: x.lower(), spam_txt)

  # Armo el data frame
  df = pd.DataFrame(ham_txt+spam_txt, columns=['text'])
  # 0 = ham, 1 = spam (es necesario binarizar para poder usar el score f_0.5)
  df['class'] = [0 for _ in range(len(ham_txt))]+[1 for _ in range(len(spam_txt))]

  print "Lei json y arme data frame"

  # Extraigo atributos simples
  # Agrego feature que clasifica los mails segun tienen o no html
  HTML = coo_matrix(map(hasHTML, df.text)).transpose()

  #) Agrego feature que clasifica los mails segun tienen o no subject
  SUBJ = coo_matrix(map(hasSubject, df.text)).transpose()

  # Longitud del mail.
  LEN = coo_matrix(map(len, df.text)).transpose()

  # Cantidad de espacios en el mail.
  SPACES = coo_matrix(map(count_spaces, df.text)).transpose()

  print "Clasifique por atributos simples"

  vectorizer = obtenerVectorizer()
  word_freq_matrix = vectorizer.transform(df.text)
  print "Arme matriz"

  X = hstack([HTML, SUBJ, LEN, SPACES, word_freq_matrix])
  y = df['class']
  select = SelectKBest(score_func=chi2, k=100)
  best_clasif = ensemble.RandomForestClassifier(max_features = 0.5, max_leaf_nodes = None, min_samples_split = 4, criterion = 'gini', n_estimators = 20)
  best_clasif_1 = BernoulliNB(binarize = 0.0, alpha = 0.25, fit_prior = False)
  best_clasif_2 = DecisionTreeClassifier(max_features = None, max_leaf_nodes = 100, min_samples_split = 2, criterion = 'gini')
  print "Univariate Selection"
  forest_res = []
  nb_res = []
  tree_res = []
  for k in [10, 100, 200, 300, 400, 500]:
    select = SelectKBest(score_func=chi2, k=k)
    X_select = select.fit_transform(X, y)
    
    #Divide into training and test-set
    X_train, X_test, y_train, y_test = train_test_split(X_select, y, test_size=0.50, random_state=42)


    best_clasif.fit(X_train,y_train)
    print "Random Forest"
    print " With k:", k
    # print "   Train:", best_clasif.score(X_train, y_train)
    score = best_clasif.score(X_test, y_test)
    print "   Test:" , score
    forest_res.append((k, score))

    best_clasif_1.fit(X_train,y_train)
    print "GaussianNB"
    print " With k:", k
    # print "   Train:", best_clasif_1.score(X_train, y_train)
    print "   Test:" , best_clasif_1.score(X_test, y_test)
    nb_res.append((k, score))

    best_clasif_2.fit(X_train,y_train)
    print "Clasif Tree"
    print " With k:", k
    # print "   Train:", best_clasif_2.score(X_train, y_train)
    print "   Test:" , best_clasif_2.score(X_test, y_test)
    tree_res.append((k, score))

  plt.plot([x for (x, y) in forest_res], [y for (x, y) in forest_res])
  plt.plot([x for (x, y) in nb_res], [y for (x, y) in nb_res])
  plt.plot([x for (x, y) in tree_res], [y for (x, y) in tree_res])

  plt.xlabel('k')
  plt.ylabel('Accuracy')

  plt.legend(["Random Forest", "GaussianNB", "Clasif Tree"], loc='lower right')

  plt.show()

raw_result = []
kbest_result = []
pca_result = []

file = open('test/raw_result.txt', 'r')
for line in file:
  raw_result.append(float(line))
file.close()

file = open('test/kbest_result.txt', 'r')
for line in file:
  kbest_result.append(float(line))
file.close()

file = open('test/pca_result.txt', 'r')
for line in file:
  pca_result.append(float(line))
file.close()

raw_result = np.array(raw_result)
kbest_result = np.array(kbest_result)

#classifier_barplot(raw_result, kbest_result, pca_result)
feature_lineplot()