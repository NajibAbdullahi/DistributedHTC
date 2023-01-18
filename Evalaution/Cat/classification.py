#!/usr/bin/env python
# coding: utf-8

# ## Classification Metrices <a class="anchor" id="16"></a>

# ### Classification Report <a class="anchor" id="16.1"></a>
#
#
# **Classification report** is another way to evaluate the classification model performance. It displays the  **precision**, **recall**, **f1** and **support** scores for the model. I have described these terms in later.
#
# We can print a classification report as follows:-

# In[1]:


from sklearn.metrics import classification_report
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)

# import libraries for plotting
import matplotlib.pyplot as plt
import seaborn as sns
# get_ipython().run_line_magic('matplotlib', 'inline')


# In[7]:


import sys
cm = pd.read_csv('/home/ubuntu/DistributedHTC/Evalaution/Cat/cm.csv')
y_test_pred = pd.read_csv('/home/ubuntu/DistributedHTC/Evalaution/Cat/y_pred_test.csv')
y_test = pd.read_csv('/home/ubuntu/DistributedHTC/Data/y_test.csv')


# In[8]:


# Convert Data to NumpyArray
y_pred_test = y_test_pred.to_numpy()
cm = cm.to_numpy()


# In[4]:


print(classification_report(y_test, y_pred_test))


# ### Classification Accuracy <a class="anchor" id="16.2"></a>

# In[9]:


TP = cm[0, 0]
TN = cm[1, 1]
FP = cm[0, 1]
FN = cm[1, 0]


# In[10]:


# print classification accuracy

classification_accuracy = (TP + TN) / float(TP + TN + FP + FN)

print('Classification accuracy : {0:0.4f}'.format(classification_accuracy))


# ### Classification Error <a class="anchor" id="16.3"></a>

# In[11]:


# print classification error

classification_error = (FP + FN) / float(TP + TN + FP + FN)

print('Classification error : {0:0.4f}'.format(classification_error))


# ### Precision <a class="anchor" id="16.4"></a>
#
#
# **Precision** can be defined as the percentage of correctly predicted positive outcomes out of all the predicted positive outcomes. It can be given as the ratio of true positives (TP) to the sum of true and false positives (TP + FP).
#
#
# So, **Precision** identifies the proportion of correctly predicted positive outcome. It is more concerned with the positive class than the negative class.
#
#
#
# Mathematically, precision can be defined as the ratio of `TP to (TP + FP).`
#
#
#

# In[12]:


# print precision score

precision = TP / float(TP + FP)


print('Precision : {0:0.4f}'.format(precision))


# ### Recall <a class="anchor" id="16.5"></a>
#
#
# Recall can be defined as the percentage of correctly predicted positive outcomes out of all the actual positive outcomes.
# It can be given as the ratio of true positives (TP) to the sum of true positives and false negatives (TP + FN). **Recall** is also called **Sensitivity**.
#
#
# **Recall** identifies the proportion of correctly predicted actual positives.
#
#
# Mathematically, recall can be given as the ratio of `TP to (TP + FN).`
#
#
#
#

# In[13]:


recall = TP / float(TP + FN)

print('Recall or Sensitivity : {0:0.4f}'.format(recall))


# ### True Positive Rate <a class="anchor" id="16.6"></a>
#
#
# **True Positive Rate** is synonymous with **Recall**.
#

# In[14]:


true_positive_rate = TP / float(TP + FN)


print('True Positive Rate : {0:0.4f}'.format(true_positive_rate))


# ### False Positive Rate <a class="anchor" id="16.7"></a>

# In[15]:


false_positive_rate = FP / float(FP + TN)


print('False Positive Rate : {0:0.4f}'.format(false_positive_rate))


# ### Specificity <a class="anchor" id="16.8"></a>

# In[16]:


specificity = TN / (TN + FP)

print('Specificity : {0:0.4f}'.format(specificity))


# ### f1-score <a class="anchor" id="16.9"></a>
#
#
# **f1-score** is the weighted harmonic mean of precision and recall. The best possible **f1-score** would be 1.0 and the worst
# would be 0.0.  **f1-score** is the harmonic mean of precision and recall. So, **f1-score** is always lower than accuracy measures as they embed precision and recall into their computation. The weighted average of `f1-score` should be used to
# compare classifier models, not global accuracy.
#
#
