#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np


# In[2]:


book=pd.read_csv("book_RS.csv", encoding='latin1')


# In[3]:


book.head


# In[4]:


book_df=pd.DataFrame(book)


# In[5]:


book_df


# In[6]:


book_df=book_df.rename({"Unnamed: 0":"index"},axis=1)


# In[7]:


book_df.head()


# In[8]:


book_df.tail()


# In[9]:


book_df.shape


# In[10]:


book_df["Book.Title"].unique()


# In[11]:


Book_title=pd.DataFrame(book_df["Book.Title"].unique())


# In[12]:


Book_title


# In[13]:


book_df["Book.Title"].value_counts()


# In[14]:


book_df["User.ID"].unique()


# In[15]:


Book_user=pd.DataFrame(book_df["User.ID"].unique())


# In[16]:


Book_user


# In[17]:


book_df.isnull().any(axis=1)


# In[18]:


book_df.isnull().any().sum()


# In[19]:


book_df.columns


# In[20]:


book_df.duplicated().sum()


# In[21]:


book_df["User.ID"].duplicated().any()


# In[22]:


book_df["User.ID"].duplicated().any()


# In[23]:


book_df[book_df["User.ID"].duplicated()]


# In[24]:


book_rat = book_df.pivot(index='index',columns='Book.Title',values='Book.Rating').reset_index(drop=True)


# In[25]:


book_rat


# In[26]:


user_book=book_df.pivot_table(index="User.ID",columns="Book.Title",values="Book.Rating").reset_index(drop=True)


# In[27]:


user_book


# In[28]:


user_book=user_book.replace(np.nan,0)


# In[29]:


user_book


# In[30]:


#Calculating Cosine Similarity between Users
from sklearn.metrics import pairwise_distances
from scipy.spatial.distance import cosine, correlation


# In[31]:


user_sim = 1 - pairwise_distances( user_book.values,metric='cosine')


# In[32]:


user_sim


# In[33]:


#Store the results in a dataframe
user_sim_df = pd.DataFrame(user_sim)


# In[34]:


#Set the index and column names to user ids 
user_sim_df.index = book_df["User.ID"].unique()
user_sim_df.columns = book_df["User.ID"].unique()


# In[35]:


user_sim_df.iloc[0:5, 0:5]


# In[36]:


np.fill_diagonal(user_sim, 0)
user_sim_df.iloc[0:5, 0:5]


# In[37]:


#Most Similar Users
user_sim_df.idxmax(axis=1)[0:5]


# In[38]:


book_df[(book_df['User.ID']==276726) | (book_df['User.ID']==276729)]


# In[39]:


user_1=book_df[book_df['User.ID']==276736]


# In[40]:


user_1["Book.Title"]


# In[41]:


user_2=book_df[book_df['User.ID']==276744]


# In[42]:


user_2["Book.Title"]


# In[43]:


#based on book title
pd.merge(user_1,user_2,on='Book.Title',how='outer')


# In[44]:


#based on ratings
pd.merge(user_1,user_2,on="Book.Rating",how='outer')


# In[45]:


pd.merge(user_1,user_2,on="Book.Rating",how='left')


# In[46]:


pd.merge(user_1,user_2,on="Book.Rating",how='right')


# # Based on ratings, recommendation...

# In[48]:


book_df.groupby('Book.Title')["Book.Rating"].mean().head()


# In[49]:


book_df.groupby('Book.Title')['Book.Rating'].mean().sort_values(ascending=False).head()


# In[50]:


book_df.groupby('Book.Title')['Book.Rating'].count().sort_values(ascending=False).head()


# In[51]:


ratings_mean_count = pd.DataFrame(book_df.groupby('Book.Title')['Book.Rating'].mean())


# In[52]:


ratings_mean_count['Ratings Count'] = pd.DataFrame(book_df.groupby('Book.Title')['Book.Rating'].count())


# In[53]:


ratings_mean_count


# In[54]:


import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('dark')
get_ipython().run_line_magic('matplotlib', 'inline')

plt.figure(figsize=(8,6))
plt.rcParams['patch.force_edgecolor'] = True
ratings_mean_count['Ratings Count'].hist(bins=20)


# In[56]:


plt.figure(figsize=(8,6))
plt.rcParams['patch.force_edgecolor'] = True
ratings_mean_count["Book.Rating"].hist(bins=50)


# In[59]:


plt.figure(figsize=(8,6))
sns.jointplot(x='Book.Rating', y='Ratings Count', data=ratings_mean_count, alpha=0.4)


# # To find Ratings for a particular book and find similar books to recommend

# In[60]:


Stardust_ratings=user_book["Stardust"]


# In[61]:


Stardust_ratings.head()


# In[62]:


#Similar books using correlation function

books_like_stardust = user_book.corrwith(Stardust_ratings)


# In[63]:


corr_Stardust = pd.DataFrame(books_like_stardust, columns=['Correlation'])


# In[64]:


corr_Stardust.head()


# In[65]:


corr_Stardust.dropna(inplace=True)
corr_Stardust.head()


# In[66]:


corr_Stardust.sort_values("Correlation",ascending=False).head(10)


# In[67]:


corr_Stardust=corr_Stardust.join(ratings_mean_count["Ratings Count"])
corr_Stardust.head(10)


# In[71]:


corr_Stardust[corr_Stardust["Ratings Count"]>3].sort_values("Correlation",ascending=False)


# In[ ]:




