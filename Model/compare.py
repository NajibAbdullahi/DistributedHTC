#!/usr/bin/env python
# coding: utf-8

# ### Compare model accuracy with null accuracy <a class="anchor" id="14.3"></a>
# 
# 
# So, the model accuracy is 0.8501. But, we cannot say that our model is very good based on the above accuracy. We must compare it with the **null accuracy**. Null accuracy is the accuracy that could be achieved by always predicting the most frequent class.
# 
# So, we should first check the class distribution in the test set. 

# In[2]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# import libraries for plotting
import matplotlib.pyplot as plt
import seaborn as sns
# get_ipython().run_line_magic('matplotlib', 'inline')


# In[3]:


import sys
X_train = pd.read_csv('/home/ubuntu/DistributedHTC/Data/X_train.csv')
X_test = pd.read_csv('/home/ubuntu/DistributedHTC/DataX_test.csv')
y_train = pd.read_csv('/home/ubuntu/DistributedHTC/Data/y_train.cs')
y_test = pd.read_csv('/home/ubuntu/DistributedHTC/Data/y_test.csv')


# In[4]:


# check class distribution in test set

y_test.value_counts()


# We can see that the occurences of most frequent class is 22067. So, we can calculate null accuracy by dividing 22067 by total number of occurences.

# In[5]:


# check null accuracy score

null_accuracy = (22067/(22067+6372))

print('Null accuracy score: {0:0.4f}'. format(null_accuracy))


# #### Interpretation
# 
# We can see that our model accuracy score is 0.8501 but null accuracy score is 0.7759. So, we can conclude that our Logistic Regression model is doing a very good job in predicting the class labels.

# #### Interpretation
# 
# Now, based on the above analysis we can conclude that our classification model accuracy is very good. Our model is doing a very good job in terms of predicting the class labels.
# 
# 
# But, it does not give the underlying distribution of values. Also, it does not tell anything about the type of errors our classifer is making. 
# 
# 
# We have another tool called `Confusion matrix` that comes to our rescue.
