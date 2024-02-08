#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns


# In[2]:


titanic_train=pd.read_csv("C:\\Users\\ritti\\Downloads\\ML_Datasets\\titanic_train.csv")


# In[3]:


titanic_train.head()


# In[4]:


titanic_train.columns


# In[5]:


titanic_train.shape


# In[6]:


titanic_train['Sex'].unique()


# In[7]:


titanic_train.isnull().sum()


# In[8]:


titanic_train['Sex']=titanic_train['Sex'].map({'male':1,'female':0})


# In[9]:


titanic_train


# In[10]:


titanic_train['Embarked'].unique()


# In[11]:


titanic_train=titanic_train.drop(['PassengerId'],axis=1)


# In[12]:


titanic_train


# In[13]:


sns.heatmap(titanic_train.isnull())


# In[ ]:





# In[14]:


sns.countplot(titanic_train,x='Pclass')


# In[15]:


sns.countplot(titanic_train,x='Survived',hue='Sex')


# In[16]:


sns.countplot(titanic_train,x='Pclass',hue='Sex')


# In[17]:


sns.countplot(titanic_train,x='Pclass',hue='Survived')


# In[18]:


sns.boxplot(titanic_train,x='Pclass',y='Age')


# In[19]:


pd.isnull(titanic_train['Age'])


# In[20]:


def fxn(cols):
    age=cols.iloc[0]
    pclass=cols.iloc[1]
    if pd.isnull(age):
        if pclass==1:
            return 37
        elif pclass==2:
            return 28
        else:
            return 24
    else:
        return age


# In[21]:


titanic_train['Age']=titanic_train[['Age','Pclass']].apply(fxn,axis=1)


# In[22]:


titanic_train


# In[23]:


sns.heatmap(titanic_train.isnull())


# In[24]:


titanic_train=titanic_train.drop(['Cabin','Name'],axis=1)


# In[25]:


titanic_train.isnull().sum()


# In[26]:


titanic_train['Embarked'].mode()


# In[27]:


titanic_train['Embarked'].fillna('S',inplace=True)


# In[28]:


titanic_train.isnull().sum()


# In[29]:


titanic_train


# In[30]:


titanic_train=titanic_train.drop(['Ticket'],axis=1)


# In[31]:


titanic_train['Embarked']=titanic_train['Embarked'].map({'S':0,'Q':1,'C':2})


# In[32]:


titanic_train['Embarked'].unique()


# In[33]:


titanic_train


# In[34]:


titanic_train.corr()


# In[35]:


X=titanic_train.iloc[:,1:]
X


# In[36]:


y=titanic_train['Survived']


# In[37]:


from sklearn.model_selection import train_test_split


# In[38]:


X_train, X_test, y_train, y_test = train_test_split(
X, y, test_size=0.20, random_state=42)


# In[39]:


from sklearn.linear_model import LogisticRegression


# In[40]:


lr=LogisticRegression(max_iter=1000)


# In[41]:


lr.fit(X_train, y_train)


# In[42]:


y_pred=lr.predict(X_test)


# In[43]:


from sklearn.metrics import accuracy_score
score=accuracy_score(y_test, y_pred)


# In[44]:


score


# In[ ]:




