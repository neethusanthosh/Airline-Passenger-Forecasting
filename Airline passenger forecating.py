#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns 


# In[16]:


df=pd.read_csv(r"E:\Machine Learning\trainingdatasets-master\international-airline-passengers.csv",skipfooter=2)


# In[17]:


df.shape


# In[18]:


df.head()


# In[19]:


df.tail()


# In[20]:


df.info()


# In[21]:


df["Month"]=pd.to_datetime(df["Month"])


# In[22]:


df.info()


# In[23]:


df.head()


# In[25]:


df1=df.drop("Month",axis=1)
df1.columns=["volume"]
df1.index=df["Month"]
df1.head()


# In[28]:


plt.plot(df1)
plt.show()


# In[33]:


rolling_mean=df1.rolling(10).mean()
rolling_std=df1.rolling(10).std()
#df1.rolling(10)
plt.plot(rolling_mean,c='r')
plt.plot(rolling_std,c='g')
plt.show()


# #dicky fuller test
# pvalue>0.05 -- null hypothesis data is not stationary
# pval<0.05 -- data is stationary

# In[34]:


from statsmodels.tsa.stattools import adfuller
adfuller(df1["volume"])


# In[35]:


dflog=np.log(df1)
plt.plot(dflog)
plt.show()


# In[36]:


dflog=dflog-dflog.shift(1)
dflog.head()


# In[37]:


rolling_mean1=dflog.rolling(10).mean()
rolling_std1=dflog.rolling(10).std()
#df1.rolling(10)
plt.plot(rolling_mean1,c='r')
plt.plot(rolling_std1,c='g')
plt.show()


# In[39]:


from statsmodels.tsa.stattools import adfuller
adfuller(dflog["volume"].dropna())


# In[ ]:




