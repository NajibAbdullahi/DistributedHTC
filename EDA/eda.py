#!/usr/bin/env python
# coding: utf-8

# ## Multivariate Analysis <a class="anchor" id="7"></a>
#
#
# - An important step in EDA is to discover patterns and relationships between variables in the dataset.
#
# - I will use heat map and pair plot to discover the patterns and relationships in the dataset.
#
# - First of all, I will draw a heat map.

# In[13]:


import sys
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)

# import libraries for plotting
import matplotlib.pyplot as plt
import seaborn as sns
# get_ipython().run_line_magic('matplotlib', 'inline')


# In[14]:


df = pd.read_csv('/home/ubuntu/DistributedHTC/Data/clean_data.csv')


# In[15]:


correlation = df.corr()


# ### Heat Map <a class="anchor" id="7.1"></a>

# In[16]:


plt.figure(figsize=(16, 12))
plt.title('Correlation Heatmap of Rain in Australia Dataset')
ax = sns.heatmap(correlation, square=True, annot=True,
                 fmt='.2f', linecolor='white')
ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
ax.set_yticklabels(ax.get_yticklabels(), rotation=30)
plt.savefig("Heatmap.png")


# #### Interpretation
#
#
#
# From the above correlation heat map, we can conclude that :-
#
# - `MinTemp` and `MaxTemp` variables are highly positively correlated (correlation coefficient = 0.74).
#
# - `MinTemp` and `Temp3pm` variables are also highly positively correlated (correlation coefficient = 0.71).
#
# - `MinTemp` and `Temp9am` variables are strongly positively correlated (correlation coefficient = 0.90).
#
# - `MaxTemp` and `Temp9am` variables are strongly positively correlated (correlation coefficient = 0.89).
#
# - `MaxTemp` and `Temp3pm` variables are also strongly positively correlated (correlation coefficient = 0.98).
#
# - `WindGustSpeed` and `WindSpeed3pm` variables are highly positively correlated (correlation coefficient = 0.69).
#
# - `Pressure9am` and `Pressure3pm` variables are strongly positively correlated (correlation coefficient = 0.96).
#
# - `Temp9am` and `Temp3pm` variables are strongly positively correlated (correlation coefficient = 0.86).
#

# ### Pair Plot <a class="anchor" id="7.2"></a>
#
#
# First of all, I will define extract the variables which are highly positively correlated.

# In[17]:


num_var = ['MinTemp', 'MaxTemp', 'Temp9am', 'Temp3pm',
           'WindGustSpeed', 'WindSpeed3pm', 'Pressure9am', 'Pressure3pm']


# Now, I will draw pairplot to depict relationship between these variables.

# In[18]:


sns.pairplot(df[num_var], kind='scatter', diag_kind='hist', palette='Rainbow')
plt.savefig("pairplot.png")


# #### Interpretation
#
#
# - I have defined a variable `num_var` which consists of `MinTemp`, `MaxTemp`, `Temp9am`, `Temp3pm`, `WindGustSpeed`, `WindSpeed3pm`, `Pressure9am` and `Pressure3pm` variables.
#
# - The above pair plot shows relationship between these variables.

# In[12]:


df.to_csv(sys.path[0] + '/../Data/clean_data.csv')
