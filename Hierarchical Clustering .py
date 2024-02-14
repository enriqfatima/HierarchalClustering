#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
from scipy import stats
from numpy.linalg import norm
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_iris 

# two useful data vizualization libraries
import matplotlib.pyplot as plt
import seaborn as sns

# setup plotting in a notebook in a reasonable way
get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('config', "InlineBackend.figure_format='retina'")

# default figure aesthetics
sns.set_style("white")
sns.set_context("notebook")


# In[ ]:


#load the iris data into a dataframe 
iris_data = load_iris() 
df = pd.DataFrame(data=iris.data,  
                  columns=iris_data.feature_names) 


# In[ ]:


#obtain a subset of the data that only contains float values...
numeric = df.columns[(df.dtypes == float)]
data = df[numeric]
data


# In[ ]:


# defines linkage function that is derived from scipy 
# https://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.hierarchy.linkage.html
def all_linkages(data):

    tree_sing = linkage(data,method='single')
    tree_cplt = linkage(data,method='complete')
    tree_avrg = linkage(data,method='average')
    tree_ward = linkage(data,method='ward')

    fig, axs = plt.subplots(2,2, figsize=(12,12))

    dendrogram(tree_sing, ax=axs[0,0])
    axs[0,0].set_title('Single linkage')
    xlim = axs[0,0].get_xlim()
    axs[0,0].set_yticks([])
    axs[0,0].set_xticks([])
    axs[0,0].set_xlim(xlim)


    dendrogram(tree_cplt, ax=axs[0,1])
    axs[0,1].set_title('Complete linkage')
    axs[0,1].set_yticks([])
    axs[0,1].set_xticks([])

    dendrogram(tree_avrg, ax=axs[1,0])
    axs[1,0].set_title('Average linkage')
    axs[1,0].set_yticks([])
    axs[1,0].set_xticks([])

    dendrogram(tree_ward, ax=axs[1,1])
    axs[1,1].set_title('Ward linkage');
    axs[1,1].set_yticks([])
    axs[1,1].set_xticks([])

    sns.despine(left=True, bottom=True)
    
    return tree_sing, tree_cplt, tree_avrg, tree_ward


# In[ ]:


tree_sing, tree_cplt, tree_avrg, tree_ward = all_linkages(data)


# In[ ]:


# Want to find a helpful tree for clustering. Thus need to 'cut' the tree. 
fig, ax = plt.subplots(figsize=(12,12))

dendrogram(tree_ward, ax=ax);
sns.despine(left=True,bottom=True)


# In[ ]:


# try different values trying to pull out unique combinations 
# fcluster takes in the tree, distance of height, and after high up the tree to cut (x-val)
clust = fcluster(tree_ward, criterion='distance', t=10000) 
clust


# In[ ]:


#df['cluster']=clust
#df.groupby('cluster')[['species','sex']].value_counts()


# In[ ]:


# Scaled Data
scaled_data = StandardScaler().fit_transform(data)
tree_sing, tree_cplt, tree_avrg, tree_ward = all_linkages(scaled_data)


# In[ ]:


fig, ax = plt.subplots(figsize=(12,12))

dendrogram(tree_ward, ax=ax);
sns.despine(left=True,bottom=True)


# In[ ]:


#clust = fcluster(tree_ward, criterion='distance', t=6.4) 

#df['cluster']=clust
#df.groupby('cluster')[['species','sex']].value_counts()

