# -*- coding: utf-8 -*- 
import json
import numpy as np
import pandas as pd
import re
from sklearn.tree import DecisionTreeClassifier
from sklearn.cross_validation import cross_val_score
from sklearn.feature_extraction.text import CountVectorizer

import sys

spam_txt= json.load(open('../dataset_json/spam_source.json'))

# Pongo todos los mails en minusculas
spam_txt = map(lambda x: x.lower(), spam_txt)

print "Lei json y puse en minusculas"

#Armo la matriz de ocurrencias (notar que es una representación para
#matriz esparsa, asi que ocupa relativamente poco)
vectorizer = CountVectorizer(token_pattern=r'[a-z]+', max_df=0.6, min_df=0.03, max_features=800, lowercase=False)
vectorizer.fit(spam_txt)
print "Prepare vectorizer"

outfile = open("vocab.txt", 'w')
for w in vectorizer.get_feature_names():
  outfile.write(w + '\n')

outfile.close()