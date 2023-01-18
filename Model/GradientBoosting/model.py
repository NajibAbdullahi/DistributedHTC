#!/usr/bin/env python
# coding: utf-8

# ## Model training <a class="anchor" id="12"></a>

# In[15]:


from sklearn.metrics import accuracy_score
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)

# import libraries for plotting
import matplotlib.pyplot as plt
import seaborn as sns
# get_ipython().run_line_magic('matplotlib', 'inline')


# In[16]:


import sys
X_train = pd.read_csv('/home/ubuntu/DistributedHTC/Data/X_train.csv')
X_test = pd.read_csv('/home/ubuntu/DistributedHTC/Data/X_test.csv')
y_train = pd.read_csv('/home/ubuntu/DistributedHTC/Data/y_train.csv')
y_test = pd.read_csv('/home/ubuntu/DistributedHTC/Data/y_test.csv')


# In[17]:


# train a logistic regression model on the training set


# instantiate the model
logreg = GradientBoostingClassifier(max_depth=2, random_state=1)


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



# In[26]:


# print the scores on training and test set




# In[28]:


# print the scores on training and test set

print('Training set score: {:.4f}'.format(logreg.score(X_train, y_train)))

print('Test set score: {:.4f}'.format(logreg.score(X_test, y_test)))



y_pred_test_ = pd.DataFrame(y_pred_test)


# In[37]:


y_pred_test_


# In[38]:


y_pred_test_.to_csv(
    '/home/ubuntu/DistributedHTC/Evalaution/GradientBoosting/y_pred_test.csv', index=False)
