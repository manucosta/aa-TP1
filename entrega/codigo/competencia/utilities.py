# -*- coding: utf-8 -*- 
import json
import numpy as np
import re
from sklearn.feature_extraction.text import CountVectorizer

def hasHTML(txt): 
  if "</html>" in txt: return 1
  return 0
def hasSubject(txt): 
  if "subject:" in txt: return 1
  return 0
def count_spaces(txt): return txt.count(" ")
def obtenerVectorizer():
  # Cargo vocabulario
  vocab_file = open("vocab.txt")
  vocab = []
  for line in vocab_file:
    vocab.append(line.strip("\n"))
  vocab_file.close()
  # Creo un vectorizador y lo devuelvo
  vectorizer = CountVectorizer(token_pattern=r'[a-z]+', vocabulary=vocab)
  return vectorizer