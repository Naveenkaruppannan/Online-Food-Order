#!/usr/bin/env python
# coding: utf-8

# In[2]:


import warnings
warnings.filterwarnings('ignore')


# In[48]:


df = pd.read_csv('onlinefoods.csv')
df.head()


# In[5]:


df.info()


# In[6]:


df.shape


# In[7]:


df.describe()


# In[51]:


df.isna().sum()


# In[8]:


plt.figure(figsize=(15, 10))
plt.title("Online Food Order Decisions Based on the Age of the Customer")
sns.histplot(x="Age", hue="Output", data=df)
plt.show()


# In[9]:


plt.figure(figsize=(15, 10))
plt.title("Online Food Order Decisions Based on the Size of the Family")
sns.histplot(x="Family size", hue="Output", data=df)
plt.show()


# In[46]:


buying_food_again = df.query("Output == 'Yes'")
buying_food_again.head()


# In[53]:


for i, cols in enumerate(df):
    print(f'{i+1} : {df[cols].value_counts()}')
    print(f'----------------------------------------')


# In[55]:


numeric_cols = df.select_dtypes(include = ['int64', 'float64']).columns
char_cols = df.select_dtypes(include = ['object']).columns


# In[56]:


f, ax = plt.subplots(5,1, figsize=(15, 15))
ax = ax.flatten()

for index, cols in enumerate(numeric_cols):
    sns.histplot(data= df, x= cols, ax = ax[index])
    ax[index].set_title(cols)
    
plt.tight_layout()
plt.show()    


# In[57]:


f, ax = plt.subplots(5,1, figsize=(15, 15))
ax = ax.flatten()

for index, cols in enumerate(numeric_cols):
    sns.boxplot(data= df, x = cols, ax = ax[index])
    ax[index].set_title(cols)
    
plt.tight_layout()
plt.show() 


# In[58]:


plt.figure(figsize=(20,20))
sns.pairplot(data= df, hue = 'Output' )
plt.show()


# In[62]:


import pandas as pd
import plotly.express as px
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix


# In[63]:


def DataTransform(df):

    df = df.drop(['Unnamed: 12'], axis=1)
    
    df['Gender'] = df['Gender'].map({"Male": 1, "Female": 0})
    df['Marital Status'] = df['Marital Status'].map({'Prefer not to say': 0, 'Single': 1, 'Married': 2})
    df['Occupation'] = df['Occupation'].map({'Student': 1, 'Employee': 2, 'Self Employeed': 3, 'House wife': 4})
    df['Educational Qualifications'] = df['Educational Qualifications'].map({'Graduate': 1, 'Post Graduate': 2,'Ph.D': 3, 'School': 4, 'Uneducated': 5})
    df['Monthly Income'] = df['Monthly Income'].map({'No Income': 0, '25001 to 50000': 50000, 'More than 50000': 70000,'10001 to 25000': 25000, 'Below Rs.10000': 10000})
    df['Output'] = df['Output'].map({'Yes': 1, 'No': 0})
    df['Feedback'] = df['Feedback'].map({'Positive': '1', 'Negative ': '0'})
    
    df = df.astype({'Output':'object'})
    
    return df

df = DataTransform(df)


# In[64]:


df.head()


# In[65]:


x = df.drop(['Feedback', 'latitude', 'longitude', 'Pin code'], axis=1)
y = df['Feedback']


# In[66]:


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)


# In[67]:


RF = RandomForestClassifier(random_state=42)
LGR = LogisticRegression()
SVM = SVC()


# In[68]:


LGR.fit(x_train, y_train)


# In[69]:


pred_LGR = LGR.predict(x_test)


# In[70]:


confusion_matrix(y_test,pred_LGR)


# In[71]:


print(f'accuracy : {accuracy_score(y_test,pred_LGR)}')


# In[72]:


RF.fit(x_train, y_train)
RandomForestClassifier(random_state=42)
pred_RF = RF.predict(x_test)
confusion_matrix(y_test,pred_RF)


# In[73]:


print(f'accuracy : {accuracy_score(y_test,pred_RF)}')


# In[ ]:




