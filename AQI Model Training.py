#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
get_ipython().run_line_magic('matplotlib', 'inline')


# In[3]:


df = pd.read_csv('AQI and Lat Long of Countries cleaned dataset.csv')


# In[4]:


df.head()


# In[16]:


df['Country'].value_counts()


# # Visualize Relationships

# In[5]:


hmap = df.corr()
sns.heatmap(hmap, annot=True, cmap="coolwarm")


# In[6]:


sns.scatterplot(x='AQI Value', y='PM2.5 AQI Value', data=df)


# In[7]:


sns.scatterplot(x='AQI Value', y='lat', data=df)


# In[8]:


sns.scatterplot(x='AQI Value', y='lng', data=df)


# In[9]:


sns.scatterplot(x='AQI Value', y='CO AQI Value', data=df)


# In[10]:


df = df[df['CO AQI Value'] < 60]


# In[11]:


sns.scatterplot(x='AQI Value', y='CO AQI Value', data=df)


# In[12]:


df = df[df['AQI Value'] < 450]


# In[13]:


sns.scatterplot(x='AQI Value', y='PM2.5 AQI Value', data=df)


# # Feature Engineering

# In[14]:


num_features = [feature for feature in df.columns if df[feature].dtype != 'O']
print('Num of Numerical features = ', len(num_features))


cat_feature = [feature for feature in df.columns if df[feature].dtype == 'O']
print('Num of categorial features = ', len(cat_feature))


descreat_feature = [feature for feature in num_features if len(df[feature].unique())<=25]
print('Num of descreat features = ', len(descreat_feature))


continous_feature = [feature for feature in num_features if len(df[feature].unique())>25]
print('Num of continous features = ', len(continous_feature))


# In[30]:


x = df.drop(['AQI Value','AQI Category'], axis = 1)
y = df['AQI Value']


# In[31]:


from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

x['Country'] = le.fit_transform(x['Country'])


# In[32]:


num_features = x.select_dtypes(exclude='object').columns
onehot_columns = ['City']
le_columns = ['Country']

from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer

num_transform = StandardScaler()
oh_transformer = OneHotEncoder(drop = 'first')

preprocessor = ColumnTransformer(
        [
            ('OneHotEncoder', oh_transformer, onehot_columns),
            ('StandardScaler', num_transform, num_features)
        ], remainder='passthrough'
)


# In[33]:


x = preprocessor.fit_transform(x)


# In[34]:


pd.DataFrame(x)


# In[35]:


from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split( x, y, test_size = 0.25, random_state = 42) 


# In[36]:


from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error


# In[37]:


def evaluate_model(true, predicted):
    mae = mean_absolute_error(true, predicted)
    mse = mean_squared_error(true, predicted)
    rmse = np.sqrt(mse)
    r2_square = r2_score(true, predicted)
    return mae, mse, rmse, r2_square


# In[38]:


models = {
    "Linear_Regressor" : LinearRegression(),
    "Lasso" : Lasso(),
    "Ridge" : Ridge(),
    "KN Regressor" : KNeighborsRegressor(),
    "Decision_Tree" : DecisionTreeRegressor(),
    "Random_Forest_Method" : RandomForestRegressor(),
    "Adaboost_Regressor" : AdaBoostRegressor()
}

for i in range(len(list(models))):
    model = list(models.values())[i]
    model.fit(x_train, y_train)
    
    y_train_pred = model.predict(x_train)
    y_test_pred = model.predict(x_test)
    
    model_train_mae, model_train_mse, model_train_rmse, model_train_r2 = evaluate_model(y_train, y_train_pred)
    model_test_mae, model_test_mse, model_test_rmse, model_test_r2 = evaluate_model(y_test, y_test_pred)

    print(list(models.keys())[i])

    print('Model performance for Training set')
    print("- Root Mean Squared Error:{:.4f}".format(model_train_rmse))
    print("- Mean Absolute Error:{:.4f}".format(model_train_mae))
    print("- R2 Score:{:.4f}".format(model_train_r2))

    print('-----------------------')

    print('Model performance for Test set')
    print("- Root Mean Squared Error: {:.4f}".format(model_test_rmse))
    print("- Mean Absolute Error:{:.4f}".format(model_test_mae))
    print("- R2 Score:{:.4f}".format(model_test_r2))

    print('='*35)
    print('\n')


# In[40]:


def map_aqi_category(aqi_value):
    if aqi_value <= 50:
        return 'Good'
    elif aqi_value <= 100:
        return 'Moderate'
    elif aqi_value <= 150:
        return 'Unhealthy for Sensitive Groups'
    elif aqi_value <= 200:
        return 'Unhealthy'
    elif aqi_value <= 300:
        return 'Very Unhealthy'
    else:
        return 'Hazardous'


# In[48]:


custom_input = {
    'Country': 'Russian Federation',
    'City': 'Praskoveya',
    'CO AQI Value': 210,
    'Ozone AQI Value': 37,
    'NO2 AQI Value': 34,
    'PM2.5 AQI Value': 15,
    'lat': 44.7444,
    'lng': 44.2031
    
}

custom_input_df = pd.DataFrame([custom_input])

custom_input_df['Country'] = le.transform(custom_input_df['Country'])

custom_input_transformed = preprocessor.transform(custom_input_df)

predicted_AQI = model.predict(custom_input_transformed)
#print(f"AQI Predicted: {predicted_AQI[0]}")

for model_name, model_instance in models.items():
    predicted_AQI = model_instance.predict(custom_input_transformed)[0]
    predicted_category = map_aqi_category(predicted_AQI)
    print(f"{model_name} - AQI Predicted: {predicted_AQI:.2f}")
    print(f"Category: {predicted_category}")


# In[27]:


df.head()


# In[49]:


import joblib

best_model = models["Random_Forest_Method"]
joblib.dump(best_model, 'best_aqi_model.pkl')
print("Model saved as 'best_aqi_model.pkl'")


# In[ ]:




