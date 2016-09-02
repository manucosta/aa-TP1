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

spam_txt = [re.sub('<[^<]+?>', '', spam) for spam in spam_txt]
spam_txt = [re.findall(r'\w+', spam) for spam in spam_txt]
print "Limpie spam"
ham_txt = [re.sub('<[^<]+?>', '', ham) for ham in ham_txt]
ham_txt = [re.findall(r'\w+', ham) for ham in ham_txt]
print "Limpie ham"

spamWords = Counter(spam_txt)
hamWords = Counter(ham_txt)

print "Conte"

print hamWords.most_common(50)
print spamWords.most_common(50)

sys.exit(0)

# Preparo data para clasificar
X = df[['html']].values
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