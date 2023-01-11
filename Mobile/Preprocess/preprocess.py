#!/usr/bin/env python
# coding: utf-8

# ### View dimensions of dataset <a class="anchor" id="4.1"></a>

# In[120]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# import libraries for plotting
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[121]:


import sys
df = pd.read_csv(sys.path[0] + '/../Data/data.csv')


# In[122]:


df.shape


# ### Preview the dataset <a class="anchor" id="4.2"></a>

# In[123]:


df.head()


# ### View column names <a class="anchor" id="4.3"></a>

# In[124]:


col_names = df.columns

col_names


# ***Drop RISK_MM variable***
# 
# It is given in the dataset description, that we should drop the RISK_MM feature variable from the dataset description. So, we should drop it as follows-

# In[125]:


df.drop(['RISK_MM'], axis=1, inplace=True)


# ### View summary of dataset <a class="anchor" id="4.5"></a>

# In[126]:


df.info()


# #### Check for missing values

# In[127]:


df['RainTomorrow'].isnull().sum()


# We can see that there are no missing values in the `RainTomorrow` target variable.

# ### Types of variables <a class="anchor" id="6.1"></a>
# 
# 
# In this section, I segregate the dataset into categorical and numerical variables. There are a mixture of categorical and numerical variables in the dataset. Categorical variables have data type object. Numerical variables have data type float64.
# 
# 
# First of all, I will find categorical variables.

# ### Explore Categorical Variables <a class="anchor" id="6.2"></a>

# In[128]:


# find categorical variables

categorical = [var for var in df.columns if df[var].dtype=='O']

print('There are {} categorical variables\n'.format(len(categorical)))

print('The categorical variables are :', categorical)


# In[129]:


# view the categorical variables

df[categorical].head()


# ### Summary of categorical variables <a class="anchor" id="6.3"></a>
# 
# 
# - There is a date variable. It is denoted by `Date` column.
# 
# 
# - There are 6 categorical variables. These are given by `Location`, `WindGustDir`, `WindDir9am`, `WindDir3pm`, `RainToday` and  `RainTomorrow`.
# 
# 
# - There are two binary categorical variables - `RainToday` and  `RainTomorrow`.
# 
# 
# - `RainTomorrow` is the target variable.

# ### Explore problems within categorical variables <a class="anchor" id="6.4"></a>
# 
# 
# First, I will explore the categorical variables.
# 
# 
# #### Missing values in categorical variables

# In[130]:


# check missing values in categorical variables

df[categorical].isnull().sum()


# In[131]:


# print categorical variables containing missing values

cat1 = [var for var in categorical if df[var].isnull().sum()!=0]

print(df[cat1].isnull().sum())


# We can see that there are only 4 categorical variables in the dataset which contains missing values. These are `WindGustDir`, `WindDir9am`, `WindDir3pm` and `RainToday`.

# #### Frequency count of categorical variables
# 
# 
# Now, I will check the frequency counts of categorical variables.

# In[132]:


# view frequency of categorical variables

for var in categorical: 
    
    print(df[var].value_counts())


# #### Number of labels: cardinality
# 
# 
# The number of labels within a categorical variable is known as **cardinality**. A high number of labels within a variable is known as **high cardinality**. High cardinality may pose some serious problems in the machine learning model. So, I will check for high cardinality.

# In[133]:


# check for cardinality in categorical variables

for var in categorical:
    
    print(var, ' contains ', len(df[var].unique()), ' labels')


# We can see that there is a `Date` variable which needs to be preprocessed. I will do preprocessing in the following section.
# 
# 
# All the other variables contain relatively smaller number of variables.

# #### Feature Engineering of Date Variable

# In[134]:


df['Date'].dtypes


# We can see that the data type of `Date` variable is object. I will parse the date currently coded as object into datetime format.

# In[135]:


# parse the dates, currently coded as strings, into datetime format

df['Date'] = pd.to_datetime(df['Date'])


# In[136]:


# extract year from date

df['Year'] = df['Date'].dt.year

df['Year'].head()


# In[137]:


# extract month from date

df['Month'] = df['Date'].dt.month

df['Month'].head()


# In[138]:


# extract day from date

df['Day'] = df['Date'].dt.day

df['Day'].head()


# In[139]:


# again view the summary of dataset

df.info()


# We can see that there are three additional columns created from `Date` variable. Now, I will drop the original `Date` variable from the dataset.

# In[140]:


# drop the original Date variable

df.drop('Date', axis=1, inplace = True)


# In[141]:


# preview the dataset again

df.head()


# Now, we can see that the `Date` variable has been removed from the dataset.
# 

# #### Explore Categorical Variables one by one
# 
# 
# Now, I will explore the categorical variables one by one. 

# In[142]:


# find categorical variables

categorical = [var for var in df.columns if df[var].dtype=='O']

print('There are {} categorical variables\n'.format(len(categorical)))

print('The categorical variables are :', categorical)


# We can see that there are 6 categorical variables in the dataset. The `Date` variable has been removed. First, I will check missing values in categorical variables.

# In[143]:


# check for missing values in categorical variables 

df[categorical].isnull().sum()


# We can see that `WindGustDir`, `WindDir9am`, `WindDir3pm`, `RainToday` variables contain missing values. I will explore these variables one by one.

# ### Explore `Location` variable

# In[144]:


# print number of labels in Location variable

print('Location contains', len(df.Location.unique()), 'labels')


# In[145]:


# check labels in location variable

df.Location.unique()


# In[146]:


# check frequency distribution of values in Location variable

df.Location.value_counts()


# In[147]:


# let's do One Hot Encoding of Location variable
# get k-1 dummy variables after One Hot Encoding 
# preview the dataset with head() method

pd.get_dummies(df.Location, drop_first=True).head()


# ### Explore `WindGustDir` variable

# In[148]:


# print number of labels in WindGustDir variable

print('WindGustDir contains', len(df['WindGustDir'].unique()), 'labels')


# In[149]:


# check labels in WindGustDir variable

df['WindGustDir'].unique()


# In[150]:


# check frequency distribution of values in WindGustDir variable

df.WindGustDir.value_counts()


# In[151]:


# let's do One Hot Encoding of WindGustDir variable
# get k-1 dummy variables after One Hot Encoding 
# also add an additional dummy variable to indicate there was missing data
# preview the dataset with head() method

pd.get_dummies(df.WindGustDir, drop_first=True, dummy_na=True).head()


# In[152]:


# sum the number of 1s per boolean variable over the rows of the dataset
# it will tell us how many observations we have for each category

pd.get_dummies(df.WindGustDir, drop_first=True, dummy_na=True).sum(axis=0)


# We can see that there are 9330 missing values in WindGustDir variable.

# In[153]:


### Explore `WindGustDir` variable

# print number of labels in WindGustDir variable

print('WindGustDir contains', len(df['WindGustDir'].unique()), 'labels')

# check labels in WindGustDir variable

df['WindGustDir'].unique()

# check frequency distribution of values in WindGustDir variable

df.WindGustDir.value_counts()

# let's do One Hot Encoding of WindGustDir variable
# get k-1 dummy variables after One Hot Encoding 
# also add an additional dummy variable to indicate there was missing data
# preview the dataset with head() method

pd.get_dummies(df.WindGustDir, drop_first=True, dummy_na=True).head()

# sum the number of 1s per boolean variable over the rows of the dataset
# it will tell us how many observations we have for each category

pd.get_dummies(df.WindGustDir, drop_first=True, dummy_na=True).sum(axis=0)


# We can see that there are 9330 missing values in WindGustDir variable.

# ### Explore `WindDir9am` variable

# In[154]:


# print number of labels in WindDir9am variable

print('WindDir9am contains', len(df['WindDir9am'].unique()), 'labels')


# In[155]:


# check labels in WindDir9am variable

df['WindDir9am'].unique()


# In[156]:


# check frequency distribution of values in WindDir9am variable

df['WindDir9am'].value_counts()


# In[157]:


# let's do One Hot Encoding of WindDir9am variable
# get k-1 dummy variables after One Hot Encoding 
# also add an additional dummy variable to indicate there was missing data
# preview the dataset with head() method

pd.get_dummies(df.WindDir9am, drop_first=True, dummy_na=True).head()


# In[158]:


# sum the number of 1s per boolean variable over the rows of the dataset
# it will tell us how many observations we have for each category

pd.get_dummies(df.WindDir9am, drop_first=True, dummy_na=True).sum(axis=0)


# We can see that there are 10013 missing values in the `WindDir9am` variable.

# ### Explore `WindDir3pm` variable

# In[159]:


# print number of labels in WindDir3pm variable

print('WindDir3pm contains', len(df['WindDir3pm'].unique()), 'labels')


# In[160]:


# check labels in WindDir3pm variable

df['WindDir3pm'].unique()


# In[161]:


# check frequency distribution of values in WindDir3pm variable

df['WindDir3pm'].value_counts()


# In[162]:


# let's do One Hot Encoding of WindDir3pm variable
# get k-1 dummy variables after One Hot Encoding 
# also add an additional dummy variable to indicate there was missing data
# preview the dataset with head() method

pd.get_dummies(df.WindDir3pm, drop_first=True, dummy_na=True).head()


# In[163]:


# sum the number of 1s per boolean variable over the rows of the dataset
# it will tell us how many observations we have for each category

pd.get_dummies(df.WindDir3pm, drop_first=True, dummy_na=True).sum(axis=0)


# There are 3778 missing values in the `WindDir3pm` variable.

# ### Explore `RainToday` variable

# In[164]:


# print number of labels in RainToday variable

print('RainToday contains', len(df['RainToday'].unique()), 'labels')


# In[165]:


# check labels in WindGustDir variable

df['RainToday'].unique()


# In[166]:


# check frequency distribution of values in WindGustDir variable

df.RainToday.value_counts()


# In[167]:


# let's do One Hot Encoding of RainToday variable
# get k-1 dummy variables after One Hot Encoding 
# also add an additional dummy variable to indicate there was missing data
# preview the dataset with head() method

pd.get_dummies(df.RainToday, drop_first=True, dummy_na=True).head()


# In[168]:


# sum the number of 1s per boolean variable over the rows of the dataset
# it will tell us how many observations we have for each category

pd.get_dummies(df.RainToday, drop_first=True, dummy_na=True).sum(axis=0)


# There are 1406 missing values in the `RainToday` variable.

# ### Explore Numerical Variables <a class="anchor" id="6.5"></a>

# In[169]:


# find numerical variables

numerical = [var for var in df.columns if df[var].dtype!='O']

print('There are {} numerical variables\n'.format(len(numerical)))

print('The numerical variables are :', numerical)


# In[170]:


# view the numerical variables

df[numerical].head()


# ### Summary of numerical variables <a class="anchor" id="6.6"></a>
# 
# 
# - There are 16 numerical variables. 
# 
# 
# - These are given by `MinTemp`, `MaxTemp`, `Rainfall`, `Evaporation`, `Sunshine`, `WindGustSpeed`, `WindSpeed9am`, `WindSpeed3pm`, `Humidity9am`, `Humidity3pm`, `Pressure9am`, `Pressure3pm`, `Cloud9am`, `Cloud3pm`, `Temp9am` and `Temp3pm`.
# 
# 
# - All of the numerical variables are of continuous type.

# ### Explore problems within numerical variables <a class="anchor" id="6.7"></a>
# 
# 
# Now, I will explore the numerical variables.
# 

# ### Missing values in numerical variables

# In[171]:


# check missing values in numerical variables

df[numerical].isnull().sum()


# We can see that all the 16 numerical variables contain missing values.

# ### Outliers in numerical variables

# In[172]:


# view summary statistics in numerical variables

print(round(df[numerical].describe()),2)


# On closer inspection, we can see that the `Rainfall`, `Evaporation`, `WindSpeed9am` and `WindSpeed3pm` columns may contain outliers.
# 
# 
# I will draw boxplots to visualise outliers in the above variables. 

# In[173]:


# draw boxplots to visualize outliers

plt.figure(figsize=(15,10))


plt.subplot(2, 2, 1)
fig = df.boxplot(column='Rainfall')
fig.set_title('')
fig.set_ylabel('Rainfall')


plt.subplot(2, 2, 2)
fig = df.boxplot(column='Evaporation')
fig.set_title('')
fig.set_ylabel('Evaporation')


plt.subplot(2, 2, 3)
fig = df.boxplot(column='WindSpeed9am')
fig.set_title('')
fig.set_ylabel('WindSpeed9am')


plt.subplot(2, 2, 4)
fig = df.boxplot(column='WindSpeed3pm')
fig.set_title('')
fig.set_ylabel('WindSpeed3pm')


# The above boxplots confirm that there are lot of outliers in these variables.

# ### Check the distribution of variables
# 
# 
# - Now, I will plot the histograms to check distributions to find out if they are normal or skewed. 
# 
# - If the variable follows normal distribution, then I will do `Extreme Value Analysis` otherwise if they are skewed, I will find IQR (Interquantile range).

# In[174]:


# plot histogram to check distribution

plt.figure(figsize=(15,10))


plt.subplot(2, 2, 1)
fig = df.Rainfall.hist(bins=10)
fig.set_xlabel('Rainfall')
fig.set_ylabel('RainTomorrow')


plt.subplot(2, 2, 2)
fig = df.Evaporation.hist(bins=10)
fig.set_xlabel('Evaporation')
fig.set_ylabel('RainTomorrow')


plt.subplot(2, 2, 3)
fig = df.WindSpeed9am.hist(bins=10)
fig.set_xlabel('WindSpeed9am')
fig.set_ylabel('RainTomorrow')


plt.subplot(2, 2, 4)
fig = df.WindSpeed3pm.hist(bins=10)
fig.set_xlabel('WindSpeed3pm')
fig.set_ylabel('RainTomorrow')


# We can see that all the four variables are skewed. So, I will use interquantile range to find outliers.

# In[175]:


# find outliers for Rainfall variable

IQR = df.Rainfall.quantile(0.75) - df.Rainfall.quantile(0.25)
Lower_fence = df.Rainfall.quantile(0.25) - (IQR * 3)
Upper_fence = df.Rainfall.quantile(0.75) + (IQR * 3)
print('Rainfall outliers are values < {lowerboundary} or > {upperboundary}'.format(lowerboundary=Lower_fence, upperboundary=Upper_fence))


# For `Rainfall`, the minimum and maximum values are 0.0 and 371.0. So, the outliers are values > 3.2.

# In[176]:


# find outliers for Evaporation variable

IQR = df.Evaporation.quantile(0.75) - df.Evaporation.quantile(0.25)
Lower_fence = df.Evaporation.quantile(0.25) - (IQR * 3)
Upper_fence = df.Evaporation.quantile(0.75) + (IQR * 3)
print('Evaporation outliers are values < {lowerboundary} or > {upperboundary}'.format(lowerboundary=Lower_fence, upperboundary=Upper_fence))


# For `Evaporation`, the minimum and maximum values are 0.0 and 145.0. So, the outliers are values > 21.8.

# In[177]:


# find outliers for WindSpeed9am variable

IQR = df.WindSpeed9am.quantile(0.75) - df.WindSpeed9am.quantile(0.25)
Lower_fence = df.WindSpeed9am.quantile(0.25) - (IQR * 3)
Upper_fence = df.WindSpeed9am.quantile(0.75) + (IQR * 3)
print('WindSpeed9am outliers are values < {lowerboundary} or > {upperboundary}'.format(lowerboundary=Lower_fence, upperboundary=Upper_fence))


# For `WindSpeed9am`, the minimum and maximum values are 0.0 and 130.0. So, the outliers are values > 55.0.

# In[178]:


# find outliers for WindSpeed3pm variable

IQR = df.WindSpeed3pm.quantile(0.75) - df.WindSpeed3pm.quantile(0.25)
Lower_fence = df.WindSpeed3pm.quantile(0.25) - (IQR * 3)
Upper_fence = df.WindSpeed3pm.quantile(0.75) + (IQR * 3)
print('WindSpeed3pm outliers are values < {lowerboundary} or > {upperboundary}'.format(lowerboundary=Lower_fence, upperboundary=Upper_fence))


# For `WindSpeed3pm`, the minimum and maximum values are 0.0 and 87.0. So, the outliers are values > 57.0.

# In[180]:


df.to_csv(sys.path[0] + '/../Data/clean_data.csv')

