#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[2]:


crime=pd.read_csv("D:\done\clustering\crime_data.csv")


# In[3]:


crime.head()


# In[4]:


def norm_func(i):
    x = (i-i.mean())/(i.std())
    return (x)


# In[5]:


df_norm = norm_func(crime.iloc[:,1:])
df_norm.head()


# In[6]:


df_norm.describe()


# In[7]:


from scipy.cluster.hierarchy import linkage 

import scipy.cluster.hierarchy as sch 


# In[9]:


z = linkage(df_norm, method="complete",metric="euclidean")


# In[30]:


plt.figure(figsize=(15, 5));
plt.title('Hierarchical Clustering Dendrogram');
plt.xlabel('Index');
plt.ylabel('Distance')
sch.dendrogram(
    z,
    leaf_rotation=0.,  # rotates the x axis labels
    leaf_font_size=8.,  # font size for the x axis labels
)
plt.show()


# In[17]:


from sklearn.cluster import AgglomerativeClustering
h=AgglomerativeClustering(n_clusters=4,linkage='complete',affinity = "euclidean").fit(df_norm) 


# In[18]:


cluster_labels=pd.Series(h.labels_)


# In[21]:


crime['clust']=cluster_labels # creating a  new column and assigning it to new column 
crime.head()


# In[22]:


from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist 


# In[23]:


model=KMeans(n_clusters=4) 
model.fit(df_norm)


# In[24]:


k = list(range(2,15))
k
TWSS = [] # variable for storing total within sum of squares for each kmeans 
for i in k:
    kmeans = KMeans(n_clusters = i)
    kmeans.fit(df_norm)
    WSS = [] # variable for storing within sum of squares for each cluster 
    for j in range(i):
        WSS.append(sum(cdist(df_norm.iloc[kmeans.labels_==j,:],kmeans.cluster_centers_[j].reshape(1,df_norm.shape[1]),"euclidean")))
    TWSS.append(sum(WSS))


# In[25]:


plt.plot(k,TWSS, 'ro-');plt.xlabel("No_of_Clusters");plt.ylabel("total_within_SS");plt.xticks(k)


# In[44]:


import seaborn as sns
plt.figure(figsize=(12,6))
sns.scatterplot(x=crime['Murder'], y = crime['Assault'],hue=h)


# In[45]:


plt.figure(figsize=(12,6))
sns.scatterplot(x=crime['Murder'], y = crime['UrbanPop'],hue=h)


# In[46]:


plt.figure(figsize=(12,6))
sns.scatterplot(x=crime['Murder'], y = crime['Rape'],hue=h)


# In[47]:


plt.figure(figsize=(12,6))
sns.scatterplot(x=crime['Assault'], y = crime['UrbanPop'],hue=h)


# In[48]:


plt.figure(figsize=(12,6))
sns.scatterplot(x=crime['Assault'], y = crime['Rape'],hue=h)


# In[49]:


plt.figure(figsize=(12,6))
sns.scatterplot(x=crime['UrbanPop'], y = crime['Rape'],hue=h)


# In[55]:


crime[crime['clust']==0]


# In[50]:


crime[crime['clust']==1]


# In[51]:


crime[crime['clust']==2]


# In[52]:


crime[crime['clust']==3]


# In[58]:


plt.hist(crime['clust'])
plt.xlabel('clusters')
plt.ylabel("distance")


# In[ ]:




