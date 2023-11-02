#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
trans=pd.read_csv('Retail_Data_Transactions.csv')
a=pd.DataFrame()


# In[2]:


trans


# In[4]:


resp=pd.read_csv('Retail_Data_Response.csv')


# In[4]:


resp


# In[5]:


df=trans.merge(resp,on='customer_id',how='left')


# In[6]:


df


# In[10]:


df.dtypes


# In[7]:


df.head()


# In[8]:


df.tail()


# In[27]:


df.describe()


# In[6]:


df.isnull().sum()


# In[11]:


df.dropna()


# In[7]:


df['trans_date']=pd.to_datetime(df['trans_date'])


# In[16]:


df


# In[46]:


df = df.fillna(value=0)


# In[47]:


df['response']=df['response'].astype('int64')


# In[23]:


df.dtypes


# In[9]:


from spicy import stats
import numpy as np

z_scores=np.abs(stats.zscore(df['tran_amount']))

threshold=3

outliers=z_scores>threshold

print(outliers)


# In[10]:


from spicy import stats
import numpy as np

z_scores=np.abs(stats.zscore(df['response']))

threshold=3

outliers=z_scores>threshold

print(outliers)


# In[11]:


import seaborn as sns
import matplotlib.pyplot as plt

sns.boxplot(x=df['tran_amount'])
plt.show()


# In[12]:


df['month']=df['trans_date'].dt.month


# In[52]:


df


# In[13]:


monthly_sales=df.groupby('month')['tran_amount'].sum()
monthly_sales=monthly_sales.sort_values(ascending=False).reset_index().head(3)
monthly_sales


# In[14]:


customer_counts=df['customer_id'].value_counts().reset_index()
customer_counts.columns=['customer_id','count']


# In[16]:


top5_cus=customer_counts.sort_values(by='count',ascending=False).head(5)
top5_cus


# In[17]:


sns.barplot(x='customer_id',y='count',data=top5_cus)


# In[20]:


customer_sales=df.groupby('customer_id')['tran_amount'].sum().reset_index()
customer_sales

top_5_sal= customer_sales.sort_values(by='tran_amount',ascending=False).head(5)
top_5_sal


# In[21]:


sns.barplot(x='customer_id',y='tran_amount',data=top_5_sal)


# In[44]:


import matplotlib.pyplot as plt
import matplotlib.dates as mdates

df['month_year'] = df['trans_date'].dt.to_period('M')
monthly_sales = df.groupby('month_year')['tran_amount'].sum()
monthly_sales.index = monthly_sales.index.to_timestamp()

plt.figure(figsize=(12,6))  
plt.plot(monthly_sales.index, monthly_sales.values)  
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))  
plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=6))  
plt.xlabel('Month-Year')
plt.ylabel('Sales')
plt.title('Monthly Sales')
plt.xticks(rotation=45) 
plt.tight_layout()  
plt.show()


# In[24]:


df


# In[56]:


recency = df.groupby('customer_id')['trans_date'].max()

frequency = df.groupby('customer_id')['trans_date'].count()

monetary = df.groupby('customer_id')['tran_amount'].sum()

df2 = pd.DataFrame({'recency': recency, 'frequency': frequency, 'monetary': monetary})


# In[49]:


recency


# In[51]:


frequency


# In[53]:


monetary


# In[57]:


df2


# In[58]:


def segment_customer(row):
    if row['recency'].year >= 2012 and row['frequency'] >= 15 and row['monetary'] > 1000:
        return 'P0'
    elif (2011 <= row['recency'].year < 2012) and (10 < row['frequency'] <= 15) and (500 < row['monetary'] <= 1000):
        return 'P1'
    else:
        return 'P2'

df2['Segment'] = df2.apply(segment_customer, axis=1)


# In[59]:


df2


# In[61]:


churn_counts = df['response'].value_counts()

churn_counts


# In[62]:


churn_counts.plot(kind='bar')


# In[63]:


top_5_customers = monetary.sort_values(ascending=False).head(5).index


top_customers_df = df[df['customer_id'].isin(top_5_customers)]

top_customers_sales = top_customers_df.groupby(['customer_id', 'month_year'])['tran_amount'].sum().unstack(level=0)
top_customers_sales.plot(kind='line')


# In[64]:


df


# In[68]:


df.to_csv('MainData.csv')


# In[67]:


df2.to_csv('AddAnlys.csv')


# In[ ]:




