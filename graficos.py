#!/usr/bin/python
# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt

def classifier_barplot():
  N = 7
  rawMeans = (20, 35, 30, 35, 27, 10, 29)
  rawStd = (2, 3, 4, 1, 2, 5, 2)

  ind = np.arange(N)  # the x locations for the groups
  width = 0.35       # the width of the bars

  fig, ax = plt.subplots()
  rects1 = ax.bar(ind, rawMeans, width, color='r', yerr=rawStd)

  selectMeans = (25, 32, 34, 20, 25)
  selectStd = (3, 5, 2, 3, 3)
  rects2 = ax.bar(ind + width, selectMeans, width, color='y', yerr=selectStd)

  # add some text for labels, title and axes ticks
  ax.set_ylabel('Scores')
  ax.set_title('Scores por clasificador y dimensionalidad')
  ax.set_xticks(ind + width)
  ax.set_xticklabels(('Decision Tree', 'Gaussian NB', 'Multinomial NB', 'Bernoulli NB', 'KNN', 'SVM', 'Random Forest'))

  ax.legend((rects1[0], rects2[0]), (u'Sin reducción de dimensionalidad', u'Con reducción de dimensionalidad'))


  def autolabel(rects):
      # attach some text labels
      for rect in rects:
          height = rect.get_height()
          ax.text(rect.get_x() + rect.get_width()/2., 1.05*height,
                  '%d' % int(height),
                  ha='center', va='bottom')

  autolabel(rects1)
  autolabel(rects2)

  plt.show()