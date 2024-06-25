#!/usr/bin/env python
# coding: utf-8

# # Used Cars Price Prediction Model

# In[174]:


# importing libraries 

import numpy as np
import pandas as pd


# In[175]:


# loading dataset

df = pd.read_csv('pakwheels.csv')


# In[176]:


# Display the first few rows of the dataset

df.head()


# In[177]:


# view all the columns of dataset

df.columns


# ## 1. Data Cleaning
# 
# - Drop irrelevant columns (ad_url, title, location, ad_last_updated, car_features, description)
# - Handle missing values if any
# - Convert price, mileage, and other categorical columns to appropriate numerical formats

# In[178]:


# Droping irrelvant columns

cars = df.drop(columns=['Unnamed: 0', 'ad_url', 'title', 'location','ad_last_updated',
       'car_features', 'description', 'body_type', 'color'])


# In[167]:


cars.head()


# In[179]:


cars.describe()


# In[168]:


cars.isnull().sum()


# ## 2. Feature Engineering:
# - Extract numercial values from columns price and mileage
# - Encoding of categoriacal varibales

# In[180]:


# convert the 'price column' to sting
cars['price'] = cars['price'].astype(str)


# In[182]:


def convert_price(price_str):
    if 'crore' in price_str:
        return float(price_str.replace('PKR ', '').replace(' crore', '')) * 10000000
    elif 'lacs' in price_str:
        return float(price_str.replace('PKR ', '').replace(' lacs', '')) * 100000
    else:
        return np.nan
    
cars['price'] = cars['price'].apply(convert_price)


# In[184]:


cars['mileage'] = cars['mileage'].str.replace(' km', '').str.replace(',', '').astype(float)


# In[190]:


cars['model_year'].dtype


# In[185]:


cars.head()


# In[152]:


# convert the mileage column to numercial 

cars['mileage'] = cars['mileage'].str.replace(' km', '').str.replace(',','')


# In[153]:


cars['mileage'] = cars['mileage'].astype(float)


# In[191]:


cars.dtypes


# In[193]:


try:
    cars['model_year'] = cars['model_year'].astype(int)
except ValueError as e:
    if 'cannot convert' in str(e).lower():
        if 'floating' in str(e).lower():
            # Convert to float first, then to integer
            cars['model_year'] = cars['model_year'].fillna(0).astype(float).astype(int)
        else:
            # Fill NaN values with a valid integer
            cars['model_year'] = cars['model_year'].fillna(0).astype(int)
    else:
        raise e


# In[197]:


cars.dtypes


# In[198]:


#convert the engine capacity column to numercial 

cars['engine_capacity'] = cars['engine_capacity'].str.replace(' cc', '')


# In[200]:


cars.head()


# In[157]:


cars.dtypes


# In[202]:


cars.isnull().sum()


# In[212]:


cars = cars.dropna()
cars.dtypes


# In[214]:


cars['engine capacity'] = cars['engine_capacity'].astype(int)


# In[217]:


zero = (cars['model_year'] == 0).sum()
zero


# In[220]:


cars_clean = cars.copy()

cars_clean = cars_clean = cars_clean[cars_clean['model_year'] != 0]


# In[223]:


cars_clean.isnull().sum()


# In[225]:


cars_clean = cars_clean.drop('engine capacity', axis=1)


# In[229]:


cars = cars_clean
cars.describe()


# ## Encoding the categorical variables 

# In[230]:


# importing the labelEncoder from scikit learn library

from sklearn.preprocessing import LabelEncoder


# In[231]:


# intialize the labelEncoder

label_encoder = LabelEncoder()


# In[232]:


categorical_columns = ['engine_type', 'transmission', 'registered_in', 'assembly']


# In[233]:


for i in categorical_columns:
    cars[i] = label_encoder.fit_transform(cars[i])


# In[234]:


cars.head()


# ## visualize and validate that regression model can be applied on your data

# In[235]:


# importing libraries

import matplotlib.pyplot as plt
import seaborn as sns


# In[236]:


for column in cars.columns:
    if column != 'price':  
        plt.figure(figsize=(6, 4))
        sns.scatterplot(x=cars[column], y=cars['price'])
        plt.title(f'Price vs {column}')
        plt.xlabel(column)
        plt.ylabel('Price')
        plt.show()


# In[237]:


corr = cars.corr()

# Plot heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(corr, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix Heatmap')
plt.show()


# In[238]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


# In[239]:


# Split the data into train and test sets

X = cars.drop(columns='price')
y = cars['price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[240]:


# Fit a linear regression model

model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)


# In[241]:


residuals = y_test - y_pred
plt.figure(figsize=(6, 4))
sns.histplot(residuals, kde=True)
plt.title('Residuals Distribution')
plt.xlabel('Residuals')
plt.ylabel('Frequency')
plt.show()


# In[242]:


plt.figure(figsize=(6, 4))
sns.scatterplot(x=y_pred, y=residuals)
plt.axhline(0, color='red', linestyle='--')
plt.title('Residuals vs Fitted Values')
plt.xlabel('Fitted Values')
plt.ylabel('Residuals')
plt.show()


# In[243]:


plt.figure(figsize=(6, 6))
sns.scatterplot(x=y_test, y=y_pred)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', linestyle='--')
plt.xlabel('Actual Prices')
plt.ylabel('Predicted Prices')
plt.title('Actual vs Predicted Prices')
plt.show()


# In[244]:


cars.describe()


# In[245]:


y_test[:5]


# In[246]:


y_pred[0:5]


# In[250]:


from sklearn.metrics import r2_score


# In[256]:


r2 = r2_score(y_test, y_pred)


# In[257]:


r2


# ## conclusion:
# 
# An R^2 value of 0.43 means that 43% of the variance in the target variable can be explained by the features included in the model, while the remaining 57% of the variance is unexplained or caused by other factors not accounted for by the model.
# 
# - Moderate predictive power: An R^2 value of 0.43 is considered moderate, meaning that the model has some predictive power, but there is still room for improvement. Generally, R^2 values above 0.6 or 0.7 are considered good, and values closer to 1 indicate a very strong predictive power.
# 
# - Potential for missing variables: The relatively low R^2 value could suggest that there are important variables or features that are not included in the model, which could potentially improve its predictive power.
# 
# - Complex or noisy data: Predicting the value of old cars can be a challenging task due to the complex nature of the problem and the potential for noisy or incomplete data. An R^2 of 0.43 may be reasonable given the inherent difficulty of the prediction task.
# 
# - Comparison to baseline: It's essential to compare the R^2 value of your model with a baseline model or a simple predictor, such as the mean or median of the target variable. If your model's R^2 is significantly higher than the baseline, it indicates that your model is providing additional predictive power.
# 
# While an R^2 of 0.43 is not exceptionally high, it still suggests that the model has some predictive value and can potentially be useful, especially if it outperforms a baseline model. However, you may want to explore ways to improve the model's performance, such as feature engineering, trying different modeling techniques, or collecting additional relevant data.

# In[ ]:




