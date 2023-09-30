#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Importing Libariries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb


# In[2]:


#Uploading Training dataset in a data frame 
df=pd.read_csv("fraudTrain.csv")
df.head(10)


# In[30]:


df.info() # Info function to check the necassary information about dataset


# In[3]:


df['is_fraud'].value_counts() #Counting the Label value which is "is_fraud" column


# In[4]:


#gender column mapping to numeric value
gender_mapping = {'M': 0, 'F': 1}
df['gender'] = df['gender'].map(gender_mapping)


# In[5]:


# Drop unnecessary columns
df.drop(["Unnamed: 0","trans_date_trans_time", "cc_num", "merchant", "first", "last", "street", "city", "state", "zip", "job", "dob", "trans_num", "unix_time"], axis=1, inplace=True)


# In[6]:


#Dividing the category in catergory wise columns
df = pd.get_dummies(df, columns=['category'], prefix=['category'])


# In[7]:


df.head()


# In[ ]:





# In[9]:


#Importing Scaling libary
from sklearn.preprocessing import StandardScaler

# Initialize the scaler
scaler = StandardScaler()

# Fit and transform the selected columns of training dataset
columns_to_scale = ['amt', 'lat', 'long', 'city_pop', 'merch_lat', 'merch_long']
df[columns_to_scale] = scaler.fit_transform(df[columns_to_scale])


# In[10]:


#Preparing Test Data 
test_data=pd.read_csv("fraudTest.csv")
#gender column mapping to numeric value
gender_mapping = {'M': 0, 'F': 1}
test_data['gender'] = test_data['gender'].map(gender_mapping)
# Drop unnecessary columns
test_data.drop(["Unnamed: 0","trans_date_trans_time", "cc_num", "merchant", "first", "last", "street", "city", "state", "zip", "job", "dob", "trans_num", "unix_time"], axis=1, inplace=True)
test_data = pd.get_dummies(test_data, columns=['category'], prefix=['category'])


# In[11]:


# Fit and transform the selected columns of testing dataset
test_data[columns_to_scale] = scaler.fit_transform(test_data[columns_to_scale])


# In[12]:


# Import RandomForestClassifier
from sklearn.ensemble import RandomForestClassifier

# Split the data into features and labels
X = df.drop("is_fraud", axis=1)
y = df["is_fraud"]


model=RandomForestClassifier()
#Train the model using fit function
model.fit(X,y)


# In[14]:


#Evalution of Model
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix


X_test = test_data.drop("is_fraud", axis=1)
y_test = test_data["is_fraud"]
# Make predictions on the test data
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred)
confusion = confusion_matrix(y_test, y_pred)

# Print or use the evaluation metrics as needed
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)
print("ROC AUC Score:", roc_auc)
print("Confusion Matrix:\n", confusion)


# In[ ]:




