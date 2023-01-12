#!/usr/bin/env python
# coding: utf-8

# ## Model training <a class="anchor" id="12"></a>

# In[15]:


from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)

# import libraries for plotting
import matplotlib.pyplot as plt
import seaborn as sns
# get_ipython().run_line_magic('matplotlib', 'inline')


# In[16]:


import sys
X_train = pd.read_csv('/home/ubuntu/DistributedHTC/Data/X_train.csv')
X_test = pd.read_csv('/home/ubuntu/DistributedHTC/DataX_test.csv')
y_train = pd.read_csv('/home/ubuntu/DistributedHTC/Data/y_train.cs')
y_test = pd.read_csv('/home/ubuntu/DistributedHTC/Data/y_test.csv')


# In[17]:


# train a logistic regression model on the training set


# instantiate the model
logreg = LogisticRegression(solver='liblinear', random_state=0)


# fit the model
logreg.fit(X_train, y_train)


# ## Predict results <a class="anchor" id="13"></a>

# In[18]:


y_pred_test = logreg.predict(X_test)

y_pred_test


# ### predict_proba method
#
#
# **predict_proba** method gives the probabilities for the target variable(0 and 1) in this case, in array form.
#
# `0 is for probability of no rain` and `1 is for probability of rain.`

# In[19]:


# probability of getting output as 0 - no rain

logreg.predict_proba(X_test)[:, 0]


# In[20]:


# probability of getting output as 1 - rain

logreg.predict_proba(X_test)[:, 1]


# ## Check accuracy score <a class="anchor" id="14"></a>

# In[21]:


print('Model accuracy score: {0:0.4f}'. format(
    accuracy_score(y_test, y_pred_test)))


# Here, **y_test** are the true class labels and **y_pred_test** are the predicted class labels in the test-set.

# ### Compare the train-set and test-set accuracy <a class="anchor" id="14.1"></a>
#
#
# Now, I will compare the train-set and test-set accuracy to check for overfitting.

# In[22]:


y_pred_train = logreg.predict(X_train)

y_pred_train


# In[23]:


print(
    'Training-set accuracy score: {0:0.4f}'. format(accuracy_score(y_train, y_pred_train)))


# ### Check for overfitting and underfitting <a class="anchor" id="14.2"></a>

# In[24]:


# print the scores on training and test set

print('Training set score: {:.4f}'.format(logreg.score(X_train, y_train)))

print('Test set score: {:.4f}'.format(logreg.score(X_test, y_test)))


# The training-set accuracy score is 0.8476 while the test-set accuracy to be 0.8501. These two values are quite comparable. So, there is no question of overfitting.
#

# In Logistic Regression, we use default value of C = 1. It provides good performance with approximately 85% accuracy on both the training and the test set. But the model performance on both the training and test set are very comparable. It is likely the case of underfitting.
#
# I will increase C and fit a more flexible model.

# In[25]:


# fit the Logsitic Regression model with C=100

# instantiate the model
logreg100 = LogisticRegression(C=100, solver='liblinear', random_state=0)


# fit the model
logreg100.fit(X_train, y_train)


# In[26]:


# print the scores on training and test set

print('Training set score: {:.4f}'.format(logreg100.score(X_train, y_train)))

print('Test set score: {:.4f}'.format(logreg100.score(X_test, y_test)))


# We can see that, C=100 results in higher test set accuracy and also a slightly increased training set accuracy. So, we can conclude that a more complex model should perform better.

# Now, I will investigate, what happens if we use more regularized model than the default value of C=1, by setting C=0.01.

# In[27]:


# fit the Logsitic Regression model with C=001

# instantiate the model
logreg001 = LogisticRegression(C=0.01, solver='liblinear', random_state=0)


# fit the model
logreg001.fit(X_train, y_train)


# In[28]:


# print the scores on training and test set

print('Training set score: {:.4f}'.format(logreg001.score(X_train, y_train)))

print('Test set score: {:.4f}'.format(logreg001.score(X_test, y_test)))


# So, if we use more regularized model by setting C=0.01, then both the training and test set accuracy decrease relative to the default parameters.

# In[36]:


y_pred_test_ = pd.DataFrame(y_pred_test)


# In[37]:


y_pred_test_


# In[38]:


y_pred_test_.to_csv(
    '/home/ubuntu/DistributedHTC/Data/y_pred_test.csv', index=False)
