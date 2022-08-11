#!/usr/bin/env python
# coding: utf-8

# # Course 7 Final Assignment
# ### Necessary Libraries 

# In[29]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler,PolynomialFeatures
from sklearn.linear_model import LinearRegression
get_ipython().run_line_magic('matplotlib', 'inline')


# In[30]:


file_name='https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-DA0101EN-SkillsNetwork/labs/FinalModule_Coursera/data/kc_house_data_NaN.csv'


# In[31]:


df=pd.read_csv(file_name)


# In[32]:


df.head(10)


# ### Question 1: Display the data types of each column using the function dtypes, then take a screenshot and submit it, include your code in the image.

# In[33]:


df.dtypes


# In[34]:


df.describe()


# ### Question 2: Drop the columns "id" and "Unnamed: 0" from axis 1 using the method drop(), then use the method describe() to obtain a statistical summary of the data. Take a screenshot and submit it, make sure the inplace parameter is set to True

# In[35]:


df.drop('Unnamed: 0', axis=1, inplace=True)
df.drop('id', axis=1, inplace=True)


# In[36]:


df.describe()


# In[37]:


print("number of NaN values for the column bedrooms :", df['bedrooms'].isnull().sum())
print("number of NaN values for the column bathrooms :", df['bathrooms'].isnull().sum())


# In[38]:


mean=df['bedrooms'].mean()
df['bedrooms'].replace(np.nan, mean, inplace=True)


# In[39]:


mean1=df['bathrooms'].mean()
df['bathrooms'].replace(np.nan, mean1, inplace=True)


# In[40]:


print('number of NaN values for the column bedrooms:', df['bedrooms'].isnull().sum())
print('number of NaN values for the column bathrooms:', df['bathrooms'].isnull().sum())


# ### Question 3: Use the method value_counts to count the number of houses with unique floor values, use the method .to_frame() to convert it to a dataframe.

# In[41]:


df['floors'].value_counts().to_frame()


# ### Question 4: Use the function boxplot in the seaborn library to determine whether houses with a waterfront view or without a waterfront view have more price outliers.

# In[42]:


sns.boxplot(x='waterfront', y='price', data=df)


# ### Question 5: Use the function regplot in the seaborn library to determine if the feature sqft_above is negatively or positively correlated with price.

# In[43]:


sns.regplot(x='sqft_above', y='price', data=df)
plt.ylim(0,)


# In[44]:


#Price and sqft_above appear to be positively correlated in the graph above


# In[45]:


df.corr()['price'].sort_values() 


# In[46]:


X=df[['long']]
Y=df[['price']]
lm=LinearRegression()
lm.fit(X,Y)
lm.score(X,Y)


# ### Question 6: Fit a linear regression model to predict the 'price' using the feature 'sqft_living' then calculate the R^2. Take a screenshot of your code and the value of the R^2.

# In[47]:


X2 = df[['sqft_living']]
lm1=LinearRegression()
lm1.fit(X2, Y)
lm1.score(X2, Y)


# ### Question 7: Fit a linear regression model to predict the 'price' using the list of features

# In[48]:


Z=df[['floors','waterfront','lat','bedrooms',\
      'sqft_basement','view','bathrooms',\
      'sqft_living15','sqft_above','grade','sqft_living']]


# In[49]:


lm2=LinearRegression()
lm2.fit(Z,Y)
lm2.score(Z,Y)


# ### Question 8: Use the list to create a pipeline object to predict the 'price', fit the object using the features in the list features, and calculate the R^2.

# In[50]:


Input=[('scale', StandardScaler()),('polynomial', PolynomialFeatures(include_bias=False)),('model', LinearRegression())]


# In[51]:


pipe=Pipeline(Input)
pipe


# In[52]:


pipe.fit(Z,Y)
pipeY=pipe.predict(Z)
pipeY[0:5]


# In[53]:


pipe.score(Z,Y)


# In[54]:


from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
print("done")


# In[55]:


features =["floors", "waterfront","lat" ,"bedrooms" ,"sqft_basement" ,"view" ,"bathrooms","sqft_living15","sqft_above","grade","sqft_living"]    
X = df[features]
Y = df['price']

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.15, random_state=1)


print("number of test samples:", x_test.shape[0])
print("number of training samples:",x_train.shape[0])


# ### Question 9: Create and fit a Ridge regression object using the training data, set the regularization parameter to 0.1, and calculate the R^2 using the test data.

# In[56]:


from sklearn.linear_model import Ridge


# In[58]:


RidgeObject=Ridge(alpha=0.1)


# In[59]:


RidgeObject.fit(x_train,y_train)


# In[60]:


RidgeObject.score(x_test,y_test)


# ### Question 10: Perform a second order polynomial transform on both the training data and testing data. Create and fit a Ridge regression object using the training data, set the regularisation parameter to 0.1, and calculate the R^2 utilising the test data provided. Take a screenshot of your code and the R^2.

# In[62]:


pr=PolynomialFeatures(degree=2)
x_train_pr=pr.fit_transform(x_train[["floors", "waterfront","lat" ,"bedrooms" ,"sqft_basement" ,"view" ,"bathrooms","sqft_living15","sqft_above","grade","sqft_living"]])
x_test_pr=pr.fit_transform(x_test[["floors", "waterfront","lat" ,"bedrooms" ,"sqft_basement" ,"view" ,"bathrooms","sqft_living15","sqft_above","grade","sqft_living"]])


# In[63]:


RidgeObject1=Ridge(alpha=0.1)
RidgeObject1.fit(x_train_pr,y_train)


# In[65]:


RidgeObject1.score(x_test_pr,y_test)


# In[66]:


jupyter notebook list


# In[ ]:




