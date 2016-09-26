from utilities import *
from scipy.sparse import coo_matrix, hstack
from sklearn.tree import DecisionTreeClassifier
from sklearn.cross_validation import cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn import ensemble
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import fbeta_score, make_scorer


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

print "Defino clasificadores"
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

mnb = MultinomialNB()
mnb_param_grid = {}

bnb = BernoulliNB()
bnb_param_grid = {}

# KNN
neigh = KNeighborsClassifier()
neigh_param_grid = {'n_neighbors': [1, 3, 5, 7, 10],
				'weights': ['distance', 'uniform'],
				'algorithm': ['ball_tree', 'kd_tree', 'brute']}

# SVM
supvecmac = svm.SVC()
supvecmac_param_grid = [
  {'C': [1, 10, 100, 1000], 'kernel': ['linear']},
  {'C': [1, 10, 100, 1000], 'gamma': [0.001, 0.0001, 0.1, 0.2, 0.5, 1.0], 'kernel': ['rbf']},
 ]

# Random Forest
ranfor = ensemble.RandomForestClassifier()
ranfor_param_grid = {"max_depth": [None, 3, 10, 100],
              "max_features": [None, 1, 3, 10, 100],
              "min_samples_split": [1, 3, 10, 100],
              "min_samples_leaf": [1, 3, 10, 100],
              "bootstrap": [True, False],
              "criterion": ["gini", "entropy"]}


clasifs = [dectree, gnb, mnb, bnb, neigh, supvecmac, ranfor]
params_grid = [dectree_param_grid, gnb_param_grid, mnb_param_grid, bnb_param_grid, neigh_param_grid, supvecmac_param_grid, ranfor_param_grid]
names = ["Decision Tree", "Gaussian Naive Bayes", "Multinomial Naive Bayes", "Bernoulli Naive Bayes", "K Nearest Neighbors", "Support Vector Machines", "Random Forest"]

clasifs = [dectree, gnb, neigh, supvecmac, ranfor]
params_grid = [dectree_param_grid, gnb_param_grid, neigh_param_grid, supvecmac_param_grid, ranfor_param_grid]
names = ["Decision Tree", "Gaussian Naive Bayes", "K Nearest Neighbors", "Support Vector Machines", "Random Forest"]

# Definimos F-Beta score con Beta=0.5
# (favorecemos precision sobre recall)
f05_scorer = make_scorer(fbeta_score, beta=0.5)
print "Empieza grid search"
# Ejecuto el clasificador entrenando con un esquema de cross validation
# de 10 folds.
for (clf, param_grid, name) in zip(clasifs, params_grid, names):
  print "Evaluando: " + name
  grid_search = GridSearchCV(clf, param_grid = param_grid, scoring = f05_scorer,cv = 10)
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

'''
print "Empiezo entrenamiento"
# Defino los clasificadores
clf = DecisionTreeClassifier(max_depth=100)
res = cross_val_score(clf, X, y, cv=10, n_jobs=1, scoring=f05_scorer)
print np.mean(res), np.std(res)
'''