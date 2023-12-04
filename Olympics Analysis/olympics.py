# Olympic Data Analysis

"""
The file athlete_events.csv contains 271116 rows and 15 columns. Each row corresponds to an individual athlete competing in an individual Olympic event (athlete-events). The columns are:

- ID - Unique number for each athlete
- Name - Athlete's name
- Sex - M or F
- Age - Integer
- Height - In centimeters
- Weight - In kilograms
- Team - Team name
- NOC - National Olympic Committee 3-letter code
- Games - Year and season
- Year - Integer
- Season - Summer or Winter
- City - Host city
- Sport - Sport
- Event - Event
- Medal - Gold, Silver, Bronze, or NA
"""

# importing the libraries
import numpy as np
import pandas as pd

# loading the dataset
df = pd.read_csv('athlete_events.csv')
region_df = pd.read_csv('noc_regions.csv')

df.tail()

df.shape
df=df[df['Season']=='Summer']
df.tail
region_df.tail()
df=df.merge(region_df,on='NOC',how='left')
df.tail()
df['region'].unique()
# preprocessing
df.isnull().sum()
df.drop_duplicates(inplace=True)
df.duplicated().sum()
df['Medal'].value_counts()

# one hot encoding
pd.get_dummies(df['Medal'])
df.shape
df=pd.concat([df,pd.get_dummies(df['Medal'])],axis=1)
df.groupby('NOC').sum()[['Gold','Silver','Bronze']].sort_values('Gold',ascending=False).reset_index()

df[(df['NOC']=='IND') & (df['Medal']=='Gold')]
# as above tally seems wrong as if a teams wins it gets medal to all players so, we convert it as country wise

medal_tally=df.drop_duplicates(subset=['Team','NOC','Games','Year','City','Sport','Event','Medal'])

medal_tally=medal_tally.groupby('region').sum()[['Gold','Silver','Bronze']].sort_values('Gold',ascending=False).reset_index()

medal_tally['total'] = medal_tally['Gold'] + medal_tally['Silver'] + medal_tally['Bronze']

year=df['Year'].unique().tolist()
year.sort()
year.insert(0,'Overall')
country=df['region'].unique().tolist()
country=np.unique(df['region'].dropna().values).tolist()
country.sort()
country.insert(0,'Overall')

def fetch_medal_tally(df,year,country):
    medal_df=df.drop_duplicates(subset=['Team','NOC','Games','Year','City','Sport','Event','Medal'])
    flag = 0
    if year == 'Overall' and country == 'Overall':
        temp_df = medal_df
    if year == 'Overall' and country != 'Overall':
        flag = 1
        temp_df = medal_df[medal_df['region']==country]
    if year != 'Overall' and country == 'Overall':
        temp_df = medal_df[medal_df['Year'] == int(year)]
    if year != 'Overall' and country != 'Overall':
        temp_df = medal_df[(medal_df['Year'] == 2016) & (medal_df['region'] == country)]
    
    if flag == 1:

        x=temp_df.groupby('Year').sum()[['Gold','Silver','Bronze']].sort_values('Year').reset_index()
    else:
        x=temp_df.groupby('region').sum()[['Gold','Silver','Bronze']].sort_values('Gold',ascending=False).reset_index()

    x['total'] = x['Gold'] + x['Silver'] + x['Bronze']

    print(x)

fetch_medal_tally(df,year=1900,country='India')
medal_df=df.drop_duplicates(subset=['Team','NOC','Games','Year','City','Sport','Event','Medal'])
medal_df[(medal_df['Year'] == 2016) & (medal_df['region'] == 'India')]

### Overall analysis
"""
- no of editions
- no of cities
- no of events / sports
- no of athletes
- participating nations
"""

df['Year'].unique().shape[0]-1
df['City'].unique().shape[0]-1
df['Sport'].unique().shape
df['Event'].unique().shape[0]
df['Name'].unique().shape
df['region'].unique().shape
df.head()

nations_over_time=df.drop_duplicates(['Year','region'])['Year'].value_counts().reset_index().sort_index(ascending=False)

import plotly.express as px
fig = px.line(nations_over_time,x='Year',y='count')
fig.show()

event_over_time=df.drop_duplicates(['Year','Event'])['Year'].value_counts().reset_index().sort_index(ascending=False)

fig = px.line(event_over_time,x='Year',y='count')
fig.show()

athlete_over_time=df.drop_duplicates(['Year','Name'])['Year'].value_counts().reset_index().sort_index(ascending=False)
px.line(athlete_over_time,x='Year',y='count')

import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(15,25))
sns.heatmap(df.pivot_table(index='Sport',columns='Year',values='Event',aggfunc='count').fillna(0).astype('int'),annot=True)

x=df.drop_duplicates(['Year','Sport','Event'])

plt.figure(figsize=(15,25))
sns.heatmap(x.pivot_table(index='Sport',columns='Year',values='Event',aggfunc='count').fillna(0).astype('int'),annot=True)

# most successful athletes
def most_successful(df,sport):
    temp_df = df.dropna(subset='Medal')

    if sport != 'Overall':
        temp_df = temp_df[temp_df['Sport'] == sport]

    x =  temp_df['Name'].value_counts().reset_index().merge(df,how='left')[['count','Name','Sport','region']].drop_duplicates('count')
    return x

most_successful(df,'Cricket')

### country wise analysis
"""
- countrywise medal tally per year (line plot)
- what countries are good at heatmap
- most successful athletes top10
"""

temp_df=df.dropna(subset=['Medal'])

temp_df.drop_duplicates(subset=['Team','NOC','Year','City','Sport','Event','Medal'],inplace=True)

new_df=temp_df[temp_df['region']=='India']
final_df=new_df.groupby('Year').count()['Medal'].reset_index()

px.line(final_df,x='Year',y='Medal')

new_df=temp_df[temp_df['region']=='UK']

plt.figure(figsize=(20,20))
sns.heatmap(new_df.pivot_table(index='Sport',columns='Year',values='Medal',aggfunc='count').fillna(0),annot=True)

# most successful athletes
def most_successful(df,country):
    temp_df = df.dropna(subset='Medal')

    
    temp_df = temp_df[temp_df['region'] == country]

    x =  temp_df['Name'].value_counts().reset_index().merge(df,how='left')[['Name','count','Sport']]
    return x

most_successful(df,'USA')

### athlete wise analysis

import plotly.figure_factory as ff
athlete_df=df.drop_duplicates(subset=['Name','region'])

x1 = athlete_df['Age'].dropna()
x2 = athlete_df[athlete_df['Medal'] == 'Gold']['Age'].dropna()
x3 = athlete_df[athlete_df['Medal'] == 'Silver']['Age'].dropna()
x4 = athlete_df[athlete_df['Medal'] == 'Bronze']['Age'].dropna()

fig=ff.create_distplot([x1,x2,x3,x4],
                       ['Overall Age','Gold Medalist','Silver medalist','Bronze medalist'],
                        show_hist=False,show_rug=False)
fig.show()

new_df['Sport']

x = []
name = []
famous_sports = ['Basketball', 'Judo', 'Football', 'Tug-Of-War', 'Athletics',
                     'Swimming', 'Badminton', 'Sailing', 'Gymnastics',
                     'Art Competitions', 'Handball', 'Weightlifting', 'Wrestling',
                     'Water Polo', 'Hockey', 'Rowing', 'Fencing',
                     'Shooting', 'Boxing', 'Taekwondo', 'Cycling', 'Diving', 'Canoeing',
                     'Tennis', 'Golf', 'Softball', 'Archery',
                     'Volleyball', 'Synchronized Swimming', 'Table Tennis', 'Baseball',
                     'Rhythmic Gymnastics', 'Rugby Sevens',
                     'Beach Volleyball', 'Triathlon', 'Rugby', 'Polo', 'Ice Hockey']

for sport in famous_sports:
    temp_df = athlete_df[athlete_df['Sport'] == sport]
    x.append(temp_df[temp_df['Medal'] == 'Gold']['Age'].dropna())
    name.append(sport)

ff.create_distplot(x,name,show_hist=False,show_rug=False)

athlete_df['Medal'].fillna('No medal',inplace=True)

sns.scatterplot(x=athlete_df['Weight'],y=athlete_df['Height'],hue=temp_df['Medal'],style=temp_df['Sex'],s=60)

men = athlete_df[athlete_df['Sex']=='M'].groupby('Year').count().reset_index()
women = athlete_df[athlete_df['Sex']=='F'].groupby('Year').count().reset_index()

final = men.merge(women,on='Year',how='left')
final.rename(columns={'Name_x':'Male','Name_y':'Female'},inplace=True)

final.fillna(0,inplace=True)

fig = px.line(final,x='Year',y=['Male','Female'])
fig.show()