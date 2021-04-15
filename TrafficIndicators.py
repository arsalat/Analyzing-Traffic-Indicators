#!/usr/bin/env python
# coding: utf-8

# 
# Indicators of Heavy Traffic on I-94
# 
# The goal of this project analyze a dataset about the westbound traffic on the I-94 Interstate highway.
# 
# We wil determine a few indicators of heavy traffic on I-94 such as :
# Weather type
# Time of the day
# Time of the week
# 
# The I-94 Traffic Dataset can be found over here : https://archive.ics.uci.edu/ml/datasets/Metro+Interstate+Traffic+Volume

# In[ ]:


# loading and examning the data
import pandas as pd

i_94 = pd.read_csv('Metro_Interstate_Traffic_Volume.csv')
i_94.head()


# In[ ]:


i_94.tail()


# In[ ]:


i_94.info()


# The dataset has 48,204 rows and 9 columns. Each row increments every hour starting from 9:00 am on 2012-10-02 to 23:00 on 2018-09-30. There are no null values.
# 
# The dataset documentation mentions that a station located approximately midway between Minneapolis and Saint Paul recorded the traffic data. Also, the station only records westbound traffic (cars moving from east to west).This means that the results of our analysis will be about the westbound traffic in the proximity of that station. In other words, we should avoid generalizing our results for the entire I-94 highway.
# 

# 
# We're going to start our analysis by examining the distribution of the traffic_volume column.

# In[ ]:


# Plot a histogram to examine the distribution of the traffic_volume column

import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
i_94['traffic_volume'].plot.hist()
plt.show()


# In[ ]:


i_94['traffic_volume'].describe()


# Based on the statistics and the histogram distribution, we will try to predict if daytime and nighttime have any influence on the traffic volume. We can see that at a certain time, 25% of cars, about 1193 travelled through the interstate. Comparing this to the max number of cars that travelled at a certain hour, 7280, we can say that this 25% travel happened at night when traffic is less and roadwork is in progress. 
# 
# To confirm our hypothesis, we'll start by dividing the dataset into two parts adn then analyze data for each interval:
# 
# Daytime data: hours from 7 AM to 7 PM (12 hours)
# Nighttime data: hours from 7 PM to 7 AM (12 hours)
# 
# 

# In[ ]:


# Transform the date_time column to datetime by using the function pd.to_datetime(). Use the Series.dt.hour property to get the hour of every instance of the date_time column and do the following:
# Isolate the daytime and nighttime data.

i_94['date_time'] = pd.to_datetime(i_94['date_time'])

day = i_94.copy()[(i_94['date_time'].dt.hour >= 7) & (i_94['date_time'].dt.hour < 19)]
print(day.shape)

night = i_94.copy()[(i_94['date_time'].dt.hour >= 19) | (i_94['date_time'].dt.hour < 7)]
print(night.shape)


# In[ ]:


# plot histograms of traffic_volume for both day and night to compare

plt.figure(figsize=(11,3.5))

plt.subplot(1, 2, 1)
plt.hist(day['traffic_volume'])
plt.xlim(-100, 7500)
plt.ylim(0, 8000)
plt.title('Traffic Volume: Day')
plt.ylabel('Frequency')
plt.xlabel('Traffic Volume')

plt.subplot(1, 2, 2)
plt.hist(night['traffic_volume'])
plt.xlim(-100, 7500)
plt.ylim(0, 8000)
plt.title('Traffic Volume: Night')
plt.ylabel('Frequency')
plt.xlabel('Traffic Volume')

plt.show()


# In[ ]:


day['traffic_volume'].describe()


# In[ ]:


night['traffic_volume'].describe()


# Both histograms are differently skewed, indicating that there is indeed a difference in traffic volume during daytime and nighttime.
# 
# During the daytime, traffic is high, as we can see that the traffic volume is left skewed. The histogram displaying the nighttime data is right skewed. This means that most of the traffic volume values are low — 75% of the time. 
# 
# Since we are only interested in heavy traffic indicators, we will only focus on the daytime data. 

# One of the possible indicators of heavy traffic is time. There might be more people on the road in a certain month, on a certain day, or at a certain time of the day.
# 
# We're going to look at a few line plots showing how the traffic volume changed according to the following parameters:
# 
# Month
# Day of the week
# Time of day

# In[ ]:


# the monthly traffic volume averages
day['month'] = day['date_time'].dt.month
by_month = day.groupby('month').mean()
by_month['traffic_volume'].plot.line()
plt.show()


# From the graph, we can predict that traffic is generally higher in the warmer months and lighter in the colder months. We also see a dip during the month of July. Let's investigate this further. 

# In[ ]:


# the daily traffic volume averages
day['dayofweek'] = day['date_time'].dt.dayofweek
by_dayofweek = day.groupby('dayofweek').mean()
by_dayofweek['traffic_volume'].plot.line()
plt.show()


# Traffic volume is significantly heavier on business days (Monday – Friday). Except for Monday, we only see values over 5,000 during business days. Traffic is lighter on weekends, with values below 4,000 cars.
# 
# 

# In[ ]:


#the yearly traffic volume averages
day['year'] = day['date_time'].dt.year
only_july = day[day['month'] == 7]
only_july.groupby('year').mean()['traffic_volume'].plot.line()
plt.show()


# Analyzing the traffic volumes yearly data reveals that traffic is heavy during July, except for the year 2016. This could've been due to construction.  

# Let's visualize how the traffic volume changes by time of the day by plotting traffic volume changes during business days and during weekends.

# In[ ]:


day['hour'] = day['date_time'].dt.hour
bussiness_days = day.copy()[day['dayofweek'] <= 4] # 4 == Friday
weekend = day.copy()[day['dayofweek'] >= 5] # 5 = Saturday
by_hour_business = bussiness_days.groupby('hour').mean()
by_hour_weekend = weekend.groupby('hour').mean()


plt.figure(figsize=(11,3.5))

plt.subplot(1, 2, 1)
by_hour_business['traffic_volume'].plot.line()
plt.xlim(6,20)
plt.ylim(1500,6500)
plt.title('Traffic Volume By Hour: Monday–Friday')

plt.subplot(1, 2, 2)
by_hour_weekend['traffic_volume'].plot.line()
plt.xlim(6,20)
plt.ylim(1500,6500)
plt.title('Traffic Volume By Hour: Weekend')

plt.show()


# 
# From the graphs we can see that 7 and 16 are the rush hours and we can see the highest traffic at these two times as people are traveling to work and back from work. The traffic volume is higher on weekdays, reaching a maximum of 6000 cars, when compared to weekends for all hours where the maximum number of cars is between 4000-5000. 
# 
# We can summarize the time indicators based on these findings:
#     The traffic is usually heavier during warm months (March–October) compared to cold months (November–February).
#     The traffic is usually heavier on weekdays compared to weekends.
# 

# Weather Indicators
# 
# Another possible indicator of heavy traffic is weather. The dataset provides us with a few useful columns about weather: temp, rain_1h, snow_1h, clouds_all, weather_main, weather_description.
# 
# let's find the correlation between these weather columns and traffic volume. 

# In[ ]:


day.corr()['traffic_volume']


# Temperature shows the strongest correlation with traffic volume (+0.13) as compared to the other columns.
# 
# Let's generate a scatter plot to visualize the correlation between temp and traffic_volume.

# In[ ]:


day.plot.scatter('traffic_volume', 'temp')
plt.ylim(230, 320) # two wrong 0K temperatures mess up the y-axis
plt.show()


# 
# We can conclude that temperature doesn't look like a solid indicator of heavy traffic.
# 
# Let's now look at the other weather-related columns: weather_main and weather_description.

# Weather Types
# To start, we're going to group the data by weather_main and look at the traffic_volume averages.

# In[ ]:



by_weather_main = day.groupby('weather_main').mean()
by_weather_main['traffic_volume'].plot.barh()
plt.show()


# 
# It looks like there's no weather type where traffic volume exceeds 5,000 cars. This makes finding a heavy traffic indicator more difficult. Let's also group by weather_description, which has a more granular weather classification.

# In[ ]:



by_weather_description = day.groupby('weather_description').mean()
by_weather_description['traffic_volume'].plot.barh(figsize=(5,10))
plt.show()


# 
# It looks like there are three weather types where traffic volume exceeds 5,000:
# 
# Shower snow
# Light rain and snow
# Proximity thunderstorm with drizzle
# It's not clear why these weather types have the highest average traffic values — this is bad weather, but not that bad. Perhaps more people take their cars out of the garage when the weather is bad instead of riding a bike or walking.
