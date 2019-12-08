#!/usr/bin/env python
# coding: utf-8

# In[20]:


#Reformat Collected Data
import datetime
import pandas as pd
import numpy as np
import os
import glob
path =r'C:\\Users\\Alan Horst\\Documents\\Wind data-20170414T171044Z-001\Wind data\\Wind Data\\All'
filenames = glob.glob(path + "/*.txt")

dfs = []
for filename in filenames:
    dfs.append(pd.read_csv(filename, infer_datetime_format=True, 
                           sep=None, engine='python', header=None,
                           skipinitialspace=True))

#Format 1

for i in (3,4):
    dfs[i][[0,1]]= dfs[i][1].str.split(expand=True)
    dfs[i].columns=['Date','Time','Wind Vel 1','Wind 
                    Vel 2','Dir 1','Dir 2','NaN']
    dfs[i] = dfs[i].drop('NaN',1)
    
#Format 2

for j in (0,2):
    dfs[j].columns=['Date','Time','Wind Vel 1','Pk 1',
                    'Wind Vel 2','Pk 2','Dir 1','Dir 2']
    dfs[j] = dfs[j][:-1]
    dfs[j]['Date']='2009/' + dfs[j]['Date'].astype(str)
    dfs[j]['Time']=dfs[j]['Time'].astype(str) + ':00'
    dfs[j]['Date']=pd.to_datetime(dfs[j]['Date'],
                                  yearfirst=True)

#Format 3

for k in (1,5,6,7,8,9,10,11,12,13,14,15,16,17,18,
          19,20,21,22,23,24,25,26,27,28,29,30,31,
          32,33,34):
    dfs[k].columns=['Date','Time','Wind Vel 1',
                    'Wind Vel 2','Dir 1','Dir 2']
    to_drop = ['Buffer Was empty','login:',
               '\x1b[1;1;H\x1b[1;1;H\x1b[2JWeatherPort WS-16',
               'Modular Weather Station','Main Menu',
               '1. Station Setup','2. Output Format',
               '3. Display Log by Hours','4. Display Log by Days'
               ,'5. Data Off-load','6. Clear Logging Memory',
               'Enter your selection [1 - 6]: \x1b[1;1;H\x1b[1;1;H\x1b[2JPlease wait up to 10 seconds for first output',
               'Data-Logger Transactions','.41','.50','02',
               '/12/2010','01.16','7.35','11.50','159.79',
               '2.80','114.26']
    dfs[k] = dfs[k][~dfs[k]['Date'].isin(to_drop)]
    dfs[k]['Date']=pd.to_datetime(dfs[k]['Date'],
                                  format='%m/%d/%Y')
pd.set_option('display.max_rows', 100)


# In[2]:


#Concatenate all data into one DataFrame
big_frame = pd.concat(dfs, ignore_index=True)
to_drop = ['0.00']
big_frame= big_frame[~big_frame['Time'].isin(to_drop)]
big_frame['Date']=pd.to_datetime(big_frame['Date'],
                                 format='%Y-%m-%d').dt.date
big_frame['Time']=pd.to_datetime(big_frame['Time'],
                                 format='%H:%M:%S').dt.time
big_frame['Date'] = big_frame.apply(lambda r: 
                                    pd.datetime.combine(r['Date']
                                                        , r['Time']), 1)
del big_frame['Time']
pd.set_option('display.max_rows', 100)
big_frame


# In[3]:


import matplotlib.pyplot as plt
big_frame['Date'] = pd.to_datetime(big_frame['Date'],
                                   format="%Y-%M-%d %H:%M:%S")


# In[21]:


big_frame['Date'].apply(lambda dt: 
                        dt.date()).groupby([big_frame['Date'].
                                            apply(lambda dt: 
                                                  dt.year)]).nunique()


# In[5]:


big_frame.shape


# In[6]:


import numpy as np
import pandas as pd
df_Data = big_frame.replace(0, np.nan)
cols_of_interest = ['Wind Vel 1', 'Wind Vel 2']
df_excel = df_Data.dropna(subset = cols_of_interest)
#writer = pd.ExcelWriter('AnemometerData2.xlsx')
#df_excel.to_excel(writer, 'Sheet 1')
#writer.save()


# In[7]:


df_excel.shape


# In[15]:


from windrose import WindroseAxes
from matplotlib import pyplot as plt
import matplotlib.cm as cm

#gb = big_frame['Date'].groupby(pd.TimeGrouper(freq='M'))
ax = WindroseAxes.from_ax()
ax.box(df_excel['Dir 1'], df_excel['Wind Vel 1'], bins=np.arange(0,24,2))
ax.set_legend()
plt.show()


# In[16]:


ax = WindroseAxes.from_ax()
ax.box(df_excel['Dir 2'], df_excel['Wind Vel 2'], bins=np.arange(0,24,2))
ax.set_legend()
plt.show()


# In[17]:


ax = WindroseAxes.from_ax()
ax.contourf(df_excel['Dir 1'], df_excel['Wind Vel 1'], 
            bins=np.arange(0, 24, 2), cmap=cm.hot)
ax.set_legend()
plt.show()


# In[18]:


ax = WindroseAxes.from_ax()
ax.contourf(df_excel['Dir 2'], df_excel['Wind Vel 2'],
            bins=np.arange(0, 24, 2), cmap=cm.hot)
ax.set_legend()
plt.show()


# In[19]:


import math
df_excel['speed_x'] = df_excel['Wind Vel 1']* np.sin(df_excel['Dir 1'] * 
                                                     math.pi / 180.0)
df_excel['speed_y'] = df_excel['Wind Vel 1'] * np.cos(df_excel['Dir 1'] * 
                                                      math.pi / 180.0)
fig, ax = plt.subplots(figsize=(8, 8), dpi=80)
x0, x1 = ax.get_xlim()
y0, y1 = ax.get_ylim()
ax.set_aspect('equal')
df_excel.plot(kind='scatter', x='speed_x', y='speed_y', alpha=0.35, ax=ax)


# In[14]:


get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
cols1 = ['Pk 1', 'Wind Vel 1']
cols2 = ['Pk 2', 'Wind Vel 2']
pd.DataFrame(df_excel[cols1]).plot(kind="density",  # Plot the distribution
                               figsize=(9,9),
                               xlim=(-1,22))
pd.DataFrame(df_excel[cols2]).plot(kind="density",  # Plot the distribution
                               figsize=(9,9),
                               xlim=(-1,22))


# In[21]:


#big_frame["Power Output"]=powercoeff(.25-.45 usually)*air density*(pi*d^2/4)*big_frame["Wind Vel 1/2"]*K
#big_frame['Power Output']=.35*1.19*(pi*2^2/4)*big_frame["Wind Vel 1/2"]*K
df_excel['Date'].apply(lambda dt: dt.date()).groupby([df_excel['Date'].
                                                      apply(lambda dt: 
                                                            dt.year)]).nunique()


# In[17]:


import math
df_excel['Power Output 1'] = .5*.35*1.19*(df_excel['Wind Vel 1']**3)
df_excel['Power Output 2'] = .5*.35*1.19*(df_excel['Wind Vel 2']**3)
df_excel['Power Output 1'].mean()
#big_frame['Power Output 2'].mean()


# In[18]:


means = df_excel.set_index('Date').groupby(pd.TimeGrouper('M')).mean()
means
#writer = pd.ExcelWriter('AnemometerMeans.xlsx')
#means.to_excel(writer, 'Sheet 1')
#writer.save()


# In[14]:


df_excel['Wind Vel 2'].mean()


# In[8]:


import os
import numpy as np
a = np.array((1,2,3))
np.savetxt('a.txt', a, newline=os.linesep)  


# In[ ]:




