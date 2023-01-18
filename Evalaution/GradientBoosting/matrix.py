#!/usr/bin/env python
# coding: utf-8

# ## 15. Confusion matrix <a class="anchor" id="15"></a>
#
#
# A confusion matrix is a tool for summarizing the performance of a classification algorithm. A confusion matrix will give us a clear picture of classification model performance and the types of errors produced by the model. It gives us a summary of correct and incorrect predictions broken down by each category. The summary is represented in a tabular form.
#
#
# Four types of outcomes are possible while evaluating a classification model performance. These four outcomes are described below:-
#
#
# **True Positives (TP)** – True Positives occur when we predict an observation belongs to a certain class and the observation actually belongs to that class.
#
#
# **True Negatives (TN)** – True Negatives occur when we predict an observation does not belong to a certain class and the observation actually does not belong to that class.
#
#
# **False Positives (FP)** – False Positives occur when we predict an observation belongs to a    certain class but the observation actually does not belong to that class. This type of error is called **Type I error.**
#
#
#
# **False Negatives (FN)** – False Negatives occur when we predict an observation does not belong to a certain class but the observation actually belongs to that class. This is a very serious error and it is called **Type II error.**
#
#
#
# These four outcomes are summarized in a confusion matrix given below.
#

# In[2]:


from sklearn.metrics import confusion_matrix
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)

# import libraries for plotting
import matplotlib.pyplot as plt
import seaborn as sns
# get_ipython().run_line_magic('matplotlib', 'inline')


# In[7]:


import sys
y_test_pred = pd.read_csv('/home/ubuntu/DistributedHTC/Evalaution/GradientBoosting/y_pred_test.csv')
y_test = pd.read_csv('/home/ubuntu/DistributedHTC/Data/y_test.csv')


# In[5]:


# Convert Data to NumpyArray
y_pred_test = y_test_pred.to_numpy()


# In[8]:


# Print the Confusion Matrix and slice it into four pieces


cm = confusion_matrix(y_test, y_pred_test)

print('Confusion matrix\n\n', cm)

print('\nTrue Positives(TP) = ', cm[0, 0])

print('\nTrue Negatives(TN) = ', cm[1, 1])

print('\nFalse Positives(FP) = ', cm[0, 1])

print('\nFalse Negatives(FN) = ', cm[1, 0])


# The confusion matrix shows `20892 + 3285 = 24177 correct predictions` and `3087 + 1175 = 4262 incorrect predictions`.
#
#
# In this case, we have
#
#
# - `True Positives` (Actual Positive:1 and Predict Positive:1) - 20892
#
#
# - `True Negatives` (Actual Negative:0 and Predict Negative:0) - 3285
#
#
# - `False Positives` (Actual Negative:0 but Predict Positive:1) - 1175 `(Type I error)`
#
#
# - `False Negatives` (Actual Positive:1 but Predict Negative:0) - 3087 `(Type II error)`

# In[9]:


# visualize confusion matrix with seaborn heatmap

cm_matrix = pd.DataFrame(data=cm, columns=['Actual Positive:1', 'Actual Negative:0'],
                         index=['Predict Positive:1', 'Predict Negative:0'])

confusion = sns.heatmap(cm_matrix, annot=True, fmt='d',
                        cmap='YlGnBu')
confusion = confusion.get_figure()
confusion.savefig('./Confusion.png')

# In[11]:


cm = pd.DataFrame(cm)


# In[12]:


cm.to_csv('/home/ubuntu/DistributedHTC/Evalaution/GradientBoosting/cm.csv', index=False)
