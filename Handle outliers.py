#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[2]:


import numpy as np


# In[3]:


import matplotlib.pyplot as plt


# In[4]:


from IPython import get_ipython


# In[5]:


get_ipython().run_line_magic('matplotlib', 'inline')


# In[14]:


df = pd.read_csv(r"D:/notes/sem 4/DS/project/modified_health.csv")


# In[15]:


df.boxplot(return_type='dict')
plt.plot()


# In[ ]:




