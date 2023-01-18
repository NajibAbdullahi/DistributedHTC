#!/usr/bin/env python
# coding: utf-8

# ## 1. Import libraries <a class="anchor" id="2"></a>
#
#
# The first step in building the model is to import the necessary libraries.

# In[15]:


import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import os

# import libraries for plotting
import matplotlib.pyplot as plt
import seaborn as sns
# get_ipython().run_line_magic('matplotlib', 'inline')


# ## 2. Import dataset <a class="anchor" id="3"></a>
#
#
# The next step is to import the dataset.

# In[17]:


data = '/home/ubuntu/DistributedHTC/Data/Data.csv'

df = pd.read_csv(data)


# In[18]:


print(df.shape)
print(os.getcwd())
