from utilities import *
from scipy.sparse import coo_matrix, hstack
from sklearn.cross_validation import cross_val_score, KFold
from sklearn import ensemble
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import fbeta_score, make_scorer
import numpy as np
import pickle
import sys

json_mails = sys.argv[1]

# Leo los mails (poner los paths correctos).
mails_txt= json.load(open(json_mails))

# Pongo todos los mails en minusculas
mails_txt = map(lambda x: x.lower(), mails_txt)

#print "Lei json y arme data frame"

# Extraigo atributos simples
# Agrego feature que clasifica los mails segun tienen o no html
HTML = coo_matrix(map(hasHTML, mails_txt)).transpose()

#) Agrego feature que clasifica los mails segun tienen o no subject
SUBJ = coo_matrix(map(hasSubject, mails_txt)).transpose()

# Longitud del mail.
LEN = coo_matrix(map(len, mails_txt)).transpose()

# Cantidad de espacios en el mail.
SPACES = coo_matrix(map(count_spaces, mails_txt)).transpose()

#print "Clasifique por atributos simples"

vectorizer = obtenerVectorizer()
word_freq_matrix = vectorizer.transform(mails_txt)
#print "Arme matriz"

X = hstack([HTML, SUBJ, LEN, SPACES, word_freq_matrix]).toarray()

clf = pickle.load( open('ranfor.pickle') )

y_predic = clf.predict(X)

for p in y_predic:
  if p == 1:
    print 'spam'
  else:
    print 'ham'
