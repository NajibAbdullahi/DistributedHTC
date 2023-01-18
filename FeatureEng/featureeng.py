#!/usr/bin/env python
# coding: utf-8

# ## Feature Engineering <a class="anchor" id="10"></a>
# 
# 
# **Feature Engineering** is the process of transforming raw data into useful features that help us to understand our model better and increase its predictive power. I will carry out feature engineering on different types of variables.
# 
# 
# First, I will display the categorical and numerical variables again separately.

# In[129]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# import libraries for plotting
import matplotlib.pyplot as plt
import seaborn as sns
# get_ipython().run_line_magic('matplotlib', 'inline')


# In[130]:


import sys
X_train=pd.read_csv('/home/ubuntu/DistributedHTC/Data/X_train.csv')
X_test=pd.read_csv('/home/ubuntu/DistributedHTC/Data/X_test.csv')
y_train=pd.read_csv('/home/ubuntu/DistributedHTC/Data/y_train.csv')
y_test=pd.read_csv('/home/ubuntu/DistributedHTC/Data/y_test.csv')


# In[131]:


# check data types in X_train

X_train.dtypes


# In[132]:


# display categorical variables

categorical = [col for col in X_train.columns if X_train[col].dtypes == 'O']

categorical


# In[133]:


# display numerical variables

numerical = [col for col in X_train.columns if X_train[col].dtypes != 'O']

numerical


# ### Engineering missing values in numerical variables <a class="anchor" id="10.1"></a>
# 
# 

# In[134]:


# check missing values in numerical variables in X_train

X_train[numerical].isnull().sum()


# In[135]:


# check missing values in numerical variables in X_test

X_test[numerical].isnull().sum()


# In[136]:


# print percentage of missing values in the numerical variables in training set

for col in numerical:
    if X_train[col].isnull().mean()>0:
        print(col, round(X_train[col].isnull().mean(),4))


# #### Assumption
# 
# 
# I assume that the data are missing completely at random (MCAR). There are two methods which can be used to impute missing values. One is mean or median imputation and other one is random sample imputation. When there are outliers in the dataset, we should use median imputation. So, I will use median imputation because median imputation is robust to outliers.
# 
# 
# I will impute missing values with the appropriate statistical measures of the data, in this case median. Imputation should be done over the training set, and then propagated to the test set. It means that the statistical measures to be used to fill missing values both in train and test set, should be extracted from the train set only. This is to avoid overfitting.

# In[137]:


# impute missing values in X_train and X_test with respective column median in X_train

for df1 in [X_train, X_test]:
    for col in numerical:
        col_median=X_train[col].median()
        df1[col].fillna(col_median, inplace=True)           
      


# In[138]:


# check again missing values in numerical variables in X_train

X_train[numerical].isnull().sum()


# In[139]:


# check missing values in numerical variables in X_test

X_test[numerical].isnull().sum()


# Now, we can see that there are no missing values in the numerical columns of training and test set.

# ### Engineering missing values in categorical variables <a class="anchor" id="10.2"></a>

# In[140]:


# print percentage of missing values in the categorical variables in training set

X_train[categorical].isnull().mean()


# In[141]:


# print categorical variables with missing data

for col in categorical:
    if X_train[col].isnull().mean()>0:
        print(col, (X_train[col].isnull().mean()))


# In[142]:


# impute missing categorical variables with most frequent value

for df2 in [X_train, X_test]:
    df2['WindGustDir'].fillna(X_train['WindGustDir'].mode()[0], inplace=True)
    df2['WindDir9am'].fillna(X_train['WindDir9am'].mode()[0], inplace=True)
    df2['WindDir3pm'].fillna(X_train['WindDir3pm'].mode()[0], inplace=True)
    df2['RainToday'].fillna(X_train['RainToday'].mode()[0], inplace=True)


# In[143]:


# check missing values in categorical variables in X_train

X_train[categorical].isnull().sum()


# In[144]:


# check missing values in categorical variables in X_test

X_test[categorical].isnull().sum()


# As a final check, I will check for missing values in X_train and X_test.

# In[145]:


# check missing values in X_train

X_train.isnull().sum()


# In[146]:


# check missing values in X_test

X_test.isnull().sum()


# We can see that there are no missing values in X_train and X_test.

# ### Engineering outliers in numerical variables <a class="anchor" id="10.3"></a>
# 
# 
# We have seen that the `Rainfall`, `Evaporation`, `WindSpeed9am` and `WindSpeed3pm` columns contain outliers. I will use top-coding approach to cap maximum values and remove outliers from the above variables.

# In[147]:


def max_value(df3, variable, top):
    return np.where(df3[variable]>top, top, df3[variable])

for df3 in [X_train, X_test]:
    df3['Rainfall'] = max_value(df3, 'Rainfall', 3.2)
    df3['Evaporation'] = max_value(df3, 'Evaporation', 21.8)
    df3['WindSpeed9am'] = max_value(df3, 'WindSpeed9am', 55)
    df3['WindSpeed3pm'] = max_value(df3, 'WindSpeed3pm', 57)


# In[148]:


X_train.Rainfall.max(), X_test.Rainfall.max()


# In[149]:


X_train.Evaporation.max(), X_test.Evaporation.max()


# In[150]:


X_train.WindSpeed9am.max(), X_test.WindSpeed9am.max()


# In[151]:


X_train.WindSpeed3pm.max(), X_test.WindSpeed3pm.max()


# In[152]:


X_train[numerical].describe()


# We can now see that the outliers in `Rainfall`, `Evaporation`, `WindSpeed9am` and `WindSpeed3pm` columns are capped.

# ### Encode categorical variables <a class="anchor" id="10.4"></a>

# In[153]:


# print categorical variables

categorical


# In[154]:


X_train[categorical].head()


# In[155]:


# encode RainToday variable

import category_encoders as ce

encoder = ce.BinaryEncoder(cols=['RainToday'])

X_train = encoder.fit_transform(X_train)

X_test = encoder.transform(X_test)


# In[156]:


X_train.head()


# We can see that two additional variables `RainToday_0` and `RainToday_1` are created from `RainToday` variable.
# 
# Now, I will create the `X_train` training set.

# In[157]:


X_train = pd.concat([X_train[numerical], X_train[['RainToday_0', 'RainToday_1']],
                     pd.get_dummies(X_train.Location), 
                     pd.get_dummies(X_train.WindGustDir),
                     pd.get_dummies(X_train.WindDir9am),
                     pd.get_dummies(X_train.WindDir3pm)], axis=1)


# In[158]:


X_train.head()


# Similarly, I will create the `X_test` testing set.

# In[159]:


X_test = pd.concat([X_test[numerical], X_test[['RainToday_0', 'RainToday_1']],
                     pd.get_dummies(X_test.Location), 
                     pd.get_dummies(X_test.WindGustDir),
                     pd.get_dummies(X_test.WindDir9am),
                     pd.get_dummies(X_test.WindDir3pm)], axis=1)


# In[160]:


X_test.head()


# We now have training and testing set ready for model building. Before that, we should map all the feature variables onto the same scale. It is called `feature scaling`. I will do it as follows.

# ## 11. Feature Scaling <a class="anchor" id="11"></a>

# In[161]:


X_train.describe()


# In[162]:


cols = X_train.columns


# In[163]:


from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

X_train = scaler.fit_transform(X_train)

X_test = scaler.transform(X_test)


# In[164]:


X_train = pd.DataFrame(X_train, columns=[cols])


# In[165]:


X_test = pd.DataFrame(X_test, columns=[cols])


# In[166]:


X_train.describe()


# We now have `X_train` dataset ready to be fed into the Logistic Regression classifier. I will do it as follows.

# In[167]:


X_train.to_csv('/home/ubuntu/DistributedHTC/Data/X_train.csv', index = False)
X_test.to_csv('/home/ubuntu/DistributedHTC/Data/X_test.csv', index = False)
y_train.to_csv('/home/ubuntu/DistributedHTC/Data/y_train.csv', index=False)
y_test.to_csv('/home/ubuntu/DistributedHTC/Data/y_test.csv', index=False)


