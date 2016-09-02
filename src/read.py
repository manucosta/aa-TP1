# Aprendizaje Automatico - DC, FCEN, UBA
# Segundo cuatrimestre 2016

import json
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.cross_validation import cross_val_score

import sys

# Leo los mails (poner los paths correctos).
ham_txt= json.load(open('../dataset_json/ham_dev.json'))
spam_txt= json.load(open('../dataset_json/spam_dev.json'))

print "************************HAM***************************"

for mail in ham_txt[0:1]:
  print mail
  print "----------------------------------------------------"

print "************************SPAM**************************"

for mail in spam_txt[0:5]:
  print mail
  print "----------------------------------------------------"