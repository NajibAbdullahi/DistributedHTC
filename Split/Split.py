#!/usr/bin/env python
# coding: utf-8

# ##  Declare feature vector and target variable <a class="anchor" id="8"></a>

# In[17]:


from sklearn.model_selection import train_test_split
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)

# import libraries for plotting
import matplotlib.pyplot as plt
import seaborn as sns
# get_ipython().run_line_magic('matplotlib', 'inline')


# In[18]:


import sys
df = pd.read_csv('/home/ubuntu/DistributedHTC/Data/clean_data.csv')


# In[19]:


X = df.drop(['RainTomorrow'], axis=1)

y = df['RainTomorrow']


# ## Split data into separate training and test set <a class="anchor" id="9"></a>

# In[20]:


# split X and y into training and testing sets


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=0)


# In[21]:


# check the shape of X_train and X_test

X_train.shape, X_test.shape, y_train.shape


# In[22]:


y_train.head(5)


# In[23]:


X_train.to_csv('/home/ubuntu/DistributedHTC/Data/X_train.csv', index = False)
X_test.to_csv('/home/ubuntu/DistributedHTC/Data/X_test.csv', index = False)
y_train.to_csv('/home/ubuntu/DistributedHTC/Data/y_train.csv', index=False)
y_test.to_csv('/home/ubuntu/DistributedHTC/Data/y_test.csv', index=False)
