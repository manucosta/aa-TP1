from utilities import *
from scipy.sparse import coo_matrix, hstack
from sklearn.tree import DecisionTreeClassifier
from sklearn.cross_validation import cross_val_score, KFold
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn import ensemble
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import fbeta_score, make_scorer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.decomposition import PCA
import numpy as np
import pickle

# Leo los mails (poner los paths correctos).
ham_txt= json.load(open('../../dataset_json/ham_dev.json'))
spam_txt= json.load(open('../../dataset_json/spam_dev.json'))

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

X = hstack([HTML, SUBJ, LEN, SPACES, word_freq_matrix]).toarray()
y = df['class']

transform = PCA(n_components=142)
transform.fit(X, y)

# Leo los mails (poner los paths correctos).
ham_txt= json.load(open('../../dataset_json/ham_test.json'))
spam_txt= json.load(open('../../dataset_json/spam_test.json'))

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

X = hstack([HTML, SUBJ, LEN, SPACES, word_freq_matrix]).toarray()
y = df['class']

f05_scorer = make_scorer(fbeta_score, beta=0.5)
names = ['dectree', 'gaussianNB', 'multinomialNB', 'bernoulliNB', 'knn', 'ranfor']

print "Predigo con raw features"
outfile = open('raw_result.txt', 'w')
for clf_name in names:
  clf = pickle.load( open('../raw_features/'+ clf_name + '.pickle') )
  res = f05_scorer(clf, X, y)
  print res
  outfile.write(str(res) + '\n')
outfile.close()

print "Predigo con k-best features"
# Selecciono atributos de forma univariada
mask = []
file = open('../select_features/mask.txt', 'r')
for line in file:
  mask.append(int(line))
file.close()
X_select = [fila[mask] for fila in X]
outfile = open('kbest_result.txt', 'w')
for clf_name in names:
  clf = pickle.load( open('../select_features/kbest-'+ clf_name + '.pickle') )
  res = f05_scorer(clf, X_select, y)
  print res
  outfile.write(str(res) + '\n')
outfile.close()

print "Predigo despues de aplicar PCA"
transform.transform(X)
outfile = open('pca_result.txt', 'w')
for clf_name in names:
  clf = pickle.load( open('../select_features/pca-'+ clf_name + '.pickle') )
  res = f05_scorer(clf, X, y)
  print res
  outfile.write(str(res) + '\n')
outfile.close()