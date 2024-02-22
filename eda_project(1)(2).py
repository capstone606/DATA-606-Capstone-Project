#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:




# Load energy data
energy_df = pd.read_csv(r"C:\Users\Admin\Downloads\energy_ds\energy_dataset.csv", parse_dates=['time'], index_col='time')

# Load weather data
weather_df = pd.read_csv(r"C:\Users\Admin\Downloads\energy_ds\weather_features.csv", parse_dates=['dt_iso'], index_col='dt_iso')


# In[3]:


# Energy data
energy_df.head()
energy_df.info()
energy_df.describe()


# In[4]:


# Drop columns with all missing values
energy_df.drop(['generation hydro pumped storage aggregated', 'forecast wind offshore eday ahead'], axis=1, inplace=True)

# Impute missing values for columns with few missing values 
energy_df.fillna(energy_df.mean(), inplace=True)


# In[ ]:





# In[5]:


# Weather data
weather_df.head()
weather_df.info()
weather_df.describe()


# 

# In[6]:


# Assuming extreme values are incorrect,limiting them to a realistic range 
weather_df['pressure'] = weather_df['pressure'].apply(lambda x: np.nan if x > 1050 or x < 980 else x)
weather_df['pressure'].fillna(method='ffill', inplace=True) # Fill missing values forward


# # merging two datasets 

# In[7]:



combined_df = pd.merge(energy_df, weather_df, left_index=True, right_index=True, how='inner')


# In[8]:


combined_df.head()


# In[9]:


plt.figure(figsize=(10, 8))
sns.heatmap(combined_df.corr(), cmap='magma', annot=False)


# In[10]:


correlation_matrix = combined_df.corr()


# In[11]:



target_correlations = correlation_matrix['price actual'].sort_values(ascending=False)


# In[12]:


print(target_correlations)


# as we can see generation fossil coal-derived gas                 
# generation fossil oil shale                         
# generation fossil peat                             
# generation geothermal                               
# generation marine                                  
# generation wind offshore columns has only zero value and they are not at all corelated to the "price actual" feature, so those columns should be dropped 

# In[13]:


combined_df.drop(['generation fossil coal-derived gas', 'generation fossil oil shale', 'generation fossil peat','generation geothermal',
'generation marine','generation wind offshore'], axis=1, inplace=True)


# In[14]:


combined_df


# In[15]:


# Convert index to DateTime 

combined_df.index = pd.to_datetime(combined_df.index, utc=True).tz_localize(None)


# # Visualizations

# In[16]:


sns.scatterplot(data=combined_df, x='temp', y='price actual', hue='weather_main')


# In[17]:


from statsmodels.tsa.seasonal import seasonal_decompose
import matplotlib.pyplot as plt

# Decomposing 'price actual' time series
decompose_result = seasonal_decompose(combined_df['price actual'], model='additive', period=365)

# Plotting the decomposed time series
decompose_result.plot()
plt.show()


# In[ ]:





# In[18]:


import plotly.express as px

fig = px.line(combined_df, x=combined_df.index, y='price actual', title='Interactive Time Series of Energy Prices')
fig.update_xaxes(title_text='Time')
fig.update_yaxes(title_text='Price Actual')
fig.show()


# In[19]:


corr = combined_df[['temp', 'wind_speed', 'pressure', 'price actual']].corr()
plt.figure(figsize=(10, 8))
sns.heatmap(corr, annot=True, cmap='coolwarm', linewidths=.5)
plt.title('Correlation Heatmap')
plt.show()


# In[20]:



energy_columns = ['generation biomass', 'generation fossil brown coal/lignite', 'generation fossil gas', 'generation solar', 'generation wind onshore']
combined_df[energy_columns] = combined_df[energy_columns].cumsum()  
combined_df[energy_columns].plot.area(figsize=(12, 8), title='Energy Generation Mix Over Time')
plt.xlabel('Time')
plt.ylabel('Cumulative Energy Generation')
plt.legend(loc='upper left')
plt.show()


# In[21]:


# Scatter plot of wind speed vs. wind generation
plt.figure(figsize=(10, 6))
plt.scatter(combined_df['wind_speed'], combined_df['generation wind onshore'], alpha=0.5)
plt.title('Wind Speed vs. Wind Energy Generation')
plt.xlabel('Wind Speed (m/s)')
plt.ylabel('Wind Energy Generation (MW)')
plt.show()


# In[22]:


weather_energy_columns = ['generation biomass', 'generation wind onshore', 'temp', 'wind_speed', 'clouds_all']
corr_matrix = combined_df[weather_energy_columns].corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Between Weather Conditions and Energy Generation')
plt.show()


# we can see that "generation biomass" and "generation wind onshore" are highly corelated 

# In[ ]:





# In[23]:


from statsmodels.tsa.seasonal import seasonal_decompose
import matplotlib.pyplot as plt


result = seasonal_decompose(combined_df['total load actual'], model='additive', period=365)
result.plot()
plt.show()


# In[ ]:





# In[25]:


# group by year
combined_df.groupby(combined_df.index.year)[['generation biomass', 'generation wind onshore', 'generation solar']].mean().plot(kind='bar', stacked=True, figsize=(10, 6))
plt.title('Average yearly Energy Generation by Source')
plt.xlabel('Year')
plt.ylabel('Average Generation (MW)')
plt.legend(title='Energy Source')


# In[26]:


#  group by month and plot
combined_df.groupby(combined_df.index.month)[['generation biomass', 'generation wind onshore', 'generation solar']].mean().plot(kind='bar', stacked=True, figsize=(10, 6))

plt.title('Average Monthly Energy Generation by Source')
plt.xlabel('Month')
plt.ylabel('Average Generation (MW)')
plt.legend(title='Energy Source')
plt.xticks(ticks=range(0, 12), labels=['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'], rotation=45)  # Adjust x-ticks to show month names
plt.show()


# ## Weather Impact on Energy Prices Line Chart

# In[27]:


plt.figure(figsize=(12, 6))
combined_df.resample('M').mean()['price actual'].plot()
plt.title('monthly Average Energy Prices')
plt.xlabel('month')
plt.ylabel('Average Price (€)')
plt.grid(True)
plt.show()


# ## Proportion of Energy Generated by Source Pie Chart

# In[28]:


energy_sources = ['generation biomass', 'generation wind onshore', 'generation solar']
energy_generation = combined_df[energy_sources].mean()
plt.figure(figsize=(8, 8))
plt.pie(energy_generation, labels=energy_sources, autopct='%1.1f%%', startangle=140)
plt.title('Proportion of Energy Generated by Source')
plt.show()


# ## Wind Speed vs. Energy Generation Scatter Plot with Regression Line

# In[29]:


plt.figure(figsize=(10, 6))
sns.regplot(x='wind_speed', y='generation wind onshore', data=combined_df, scatter_kws={'alpha':0.5})
plt.title('Wind Speed vs. Wind Energy Generation with Regression Line')
plt.xlabel('Wind Speed (m/s)')
plt.ylabel('Wind Energy Generation (MW)')
plt.show()


# ## Bar Graph of Average Energy Generation by Hour of Day

# In[30]:


combined_df['hour'] = combined_df.index.hour
energy_generation_hourly = combined_df.groupby('hour')[['generation biomass', 'generation wind onshore', 'generation solar']].mean()

energy_generation_hourly.plot(kind='bar', stacked=True, figsize=(12, 8))
plt.title('Average Energy Generation by Hour of Day')
plt.xlabel('Hour of Day')
plt.ylabel('Average Generation (MW)')
plt.legend(title='Energy Source')
plt.show()


# In[ ]:




