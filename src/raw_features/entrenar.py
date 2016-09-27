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

# Definimos F-Beta score con Beta=0.5
# (favorecemos precision sobre recall)
f05_scorer = make_scorer(fbeta_score, beta=0.5)

print "Defino clasificadores"
# Decision Tree
print "Decision Tree"
clf = DecisionTreeClassifier(max_features = None, max_leaf_nodes = 100, min_samples_split = 2, criterion = 'gini')
kf = KFold(72000, n_folds=10, shuffle=True)
best_score = 0
best_clf = 0
for train_index, test_index in kf:
  print("TRAIN:", train_index, "TEST:", test_index)
  X_train, X_test = X[train_index], X[test_index]
  y_train, y_test = y[train_index], y[test_index]
  clf.fit(X_train, y_train)
  score = f05_scorer(clf, X_test, y_test)
  if score > best_score:
    best_clf = clf
    best_score = score

fout = open('dectree.pickle','w')
pickle.dump(best_clf,fout)
fout.close()
'''
# Random Forest
print "Random Forest"
clf = ensemble.RandomForestClassifier(max_features = 0.5, max_leaf_nodes = None, min_samples_split = 4, criterion = 'gini', n_estimators = 20)

kf = KFold(72000, n_folds=10, shuffle=True)
best_score = 0
best_clf = 0
for train_index, test_index in kf:
  print("TRAIN:", train_index, "TEST:", test_index)
  X_train, X_test = X[train_index], X[test_index]
  y_train, y_test = y[train_index], y[test_index]
  clf.fit(X_train, y_train)
  score = f05_scorer(clf, X_test, y_test)
  if score > best_score:
    best_clf = clf
    best_score = score

fout = open('ranfor.pickle','w')
pickle.dump(clf,fout)
fout.close()

# SVM
print "SVM"
clf = svm.SVC(kernel = 'rbf', C = 1, gamma = 1.0)

kf = KFold(72000, n_folds=10, shuffle=True)
best_score = 0
best_clf = 0
for train_index, test_index in kf:
  print("TRAIN:", train_index, "TEST:", test_index)
  X_train, X_test = X[train_index], X[test_index]
  y_train, y_test = y[train_index], y[test_index]
  clf.fit(X_train, y_train)
  score = f05_scorer(clf, X_test, y_test)
  if score > best_score:
    best_clf = clf
    best_score = score

fout = open('svm.pickle','w')
pickle.dump(clf,fout)
fout.close()

# Naive Bayes
print "Gaussian NB"
clf = GaussianNB()

kf = KFold(72000, n_folds=10, shuffle=True)
best_score = 0
best_clf = 0
for train_index, test_index in kf:
  print("TRAIN:", train_index, "TEST:", test_index)
  X_train, X_test = X[train_index], X[test_index]
  y_train, y_test = y[train_index], y[test_index]
  clf.fit(X_train, y_train)
  score = f05_scorer(clf, X_test, y_test)
  if score > best_score:
    best_clf = clf
    best_score = score

fout = open('gaussianNB.pickle','w')
pickle.dump(clf,fout)
fout.close()

#########################
print "Multinomial NB"
clf = MultinomialNB(alpha = 0.25, fit_prior = False)

kf = KFold(72000, n_folds=10, shuffle=True)
best_score = 0
best_clf = 0
for train_index, test_index in kf:
  print("TRAIN:", train_index, "TEST:", test_index)
  X_train, X_test = X[train_index], X[test_index]
  y_train, y_test = y[train_index], y[test_index]
  clf.fit(X_train, y_train)
  score = f05_scorer(clf, X_test, y_test)
  if score > best_score:
    best_clf = clf
    best_score = score

fout = open('multinomialNB.pickle','w')
pickle.dump(clf,fout)
fout.close()

#######################
print "Bernoulli NB"
clf = BernoulliNB(binarize = 0.0, alpha = 0.25, fit_prior = False)

kf = KFold(72000, n_folds=10, shuffle=True)
best_score = 0
best_clf = 0
for train_index, test_index in kf:
  print("TRAIN:", train_index, "TEST:", test_index)
  X_train, X_test = X[train_index], X[test_index]
  y_train, y_test = y[train_index], y[test_index]
  clf.fit(X_train, y_train)
  score = f05_scorer(clf, X_test, y_test)
  if score > best_score:
    best_clf = clf
    best_score = score

fout = open('bernoulliNB.pickle','w')
pickle.dump(clf,fout)
fout.close()

# KNN
print "KNN"
clf = KNeighborsClassifier(n_neighbors = 1, weights = 'uniform', leaf_size = 15, algorithm = 'kd_tree')

kf = KFold(72000, n_folds=10, shuffle=True)
best_score = 0
best_clf = 0
for train_index, test_index in kf:
  print("TRAIN:", train_index, "TEST:", test_index)
  X_train, X_test = X[train_index], X[test_index]
  y_train, y_test = y[train_index], y[test_index]
  clf.fit(X_train, y_train)
  score = f05_scorer(clf, X_test, y_test)
  if score > best_score:
    best_clf = clf
    best_score = score

fout = open('knn.pickle','w')
pickle.dump(clf,fout)
fout.close()
'''