#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
get_ipython().run_line_magic('matplotlib', 'inline')


# In[4]:


data = pd.read_csv("AQI and Lat Long of Countries.csv")

df = pd.read_csv("AQI and Lat Long of Countries.csv")


# In[3]:


data.head()


# In[5]:


data.info()


# In[7]:


data.isnull().sum()


# In[9]:


data.shape


# In[10]:


data = data.dropna(subset = ['Country'])


# In[11]:


data.isnull().sum()


# In[12]:


df.applymap(lambda x: isinstance(x, str) and (x.startswith(' ') or x.endswith(' ')))


# In[14]:


data.head()


# In[15]:


data.drop('CO AQI Category', axis = 1, inplace = True)
data.drop('Ozone AQI Category', axis = 1, inplace = True)
data.drop('NO2 AQI Category', axis = 1, inplace = True)
data.drop('PM2.5 AQI Category', axis = 1, inplace = True)


# In[16]:


data.head()


# In[17]:


df.head()


# In[18]:


df = data
df.head()


# In[19]:


df.to_csv('AQI and Lat Long of Countries cleaned dataset.csv',index=False)


# In[20]:


datas = pd.read_csv('AQI and Lat Long of Countries cleaned dataset.csv')


# In[21]:


datas.head()


# In[ ]:




