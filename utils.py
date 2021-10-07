import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
import random

from sklearn.model_selection import KFold, RandomizedSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error
import lightgbm as lgb

def half_heatmap(datacorr):
    #plot heatmap mask half
    #Set and compute the Correlation Matrix:
    sns.set(style="white")
    corr = datacorr.corr()

    #Generate a mask for the upper triangle:
    mask = np.zeros_like(corr, dtype=np.bool)
    mask[np.triu_indices_from(mask)] = True
    #Set up the matplotlib figure and a diverging colormap:
    f, ax = plt.subplots(figsize=(18, 15))
    cmap = sns.diverging_palette(220, 10, as_cmap=True)
    #Draw the heatmap with the mask and correct aspect ratio:
    sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0, annot=True, 
    square=True, linewidths=.5, cbar_kws={"shrink": .5})
    return

#Histogram:
def histplot(dataset2):
    fig = plt.figure(figsize=(15, 12))
    plt.suptitle('Histograms of Numerical Columns\n',horizontalalignment="center",fontstyle = "normal", fontsize = 14, fontfamily = "sans-serif")
    for i in range(dataset2.shape[1]):
        plt.subplot(6, 3, i + 1)
        f = plt.gca()
        f.set_title(dataset2.columns.values[i])

        vals = np.size(dataset2.iloc[:, i].unique())
        if vals >= 100:
            vals = 100

        plt.hist(dataset2.iloc[:, i], bins=vals, color = '#ec838a')
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        return

# Plot overview
def plot_overview(df, column, top_count=5):
    agg_func = {'NUM_CONVOS':['count'],
                   'TOTAL_ACTIVITY_MINUTES2':['mean'],
                   'NUM_ALL_MEALS':['mean'],
                   'NUM_ACTIVITY_MEASUREMENTS' : ['mean'],
                   'NUM_WEIGHTS': ['mean'],
                    'NUM_NOTIFS': ['mean']
               } #'revenue_per_guest':['mean']
    temp_df = df.groupby(column).agg(agg_func)
    temp_df.columns = ['_'.join(col)for col in temp_df.columns.values]
    temp_df = temp_df.sort_values(by='PST_DATE', ascending=False)
    temp_df.reset_index(inplace=True)
    if len(temp_df)>top_count:
        temp_df = temp_df.loc[:top_count-1,:]
        
    temp_df = temp_df.sort_values(by='PST_DATE', ascending=False)
    
     # Plot count and price
    fig1 = plt.figure(figsize=(10,6))
    ax1 = fig1.add_subplot(211)
    ax12 = ax1.twinx()
    temp_df.plot(x=column, y='id_count', kind='bar', color='blue', ax=ax1, width=0.4, position=1, legend=False)
    temp_df.plot(x=column, y='price_per_person_mean', kind='bar', color='red',ax=ax12, width=0.4, position=0, legend=False)
    #temp_df['id_count'].plot(kind='bar', color='blue', ax=ax1, width=0.4, position=1)
    #temp_df['price_mean'].plot(kind='bar', color='red',ax=ax12, width=0.4, position=0)
    ax1.set_ylabel('Count', color='blue')
    ax1.tick_params(axis='y', labelcolor='blue', labelbottom=False)
    ax1.tick_params(axis='x', which='both', bottom=False, labelbottom=False)
    ax12.set_ylabel('Price per person ($)', color='red')
    ax12.tick_params(axis='y', labelcolor='red', labelbottom=False)
    ax12.tick_params(axis='x', which='both', bottom=False, labelbottom=False)
    ax1.set_xlim(left=-.6)

    ax2 = fig1.add_subplot(212)
    ax22 = ax2.twinx()
    temp_df.plot(x=column, y='booking_rate(%)_mean', kind='bar', color='blue', ax=ax2, width=0.4, position=1, legend=False)
    #temp_df.plot(x=column, y='revenue_per_guest_mean', kind='bar', color='red',ax=ax22, width=0.25, position=0)
    temp_df.plot(x=column, y='daily_revenue_mean', kind='bar', color='red',ax=ax22, width=0.4, position=0, legend=False)
    ax2.set_ylabel('Booking rate(%)', color='blue')
    ax2.tick_params(axis='y', labelcolor='blue', labelbottom=False)
    ax2.tick_params(axis='x', rotation=30)
    ax22.set_ylabel('Daily revenue ($)', color='red')
    ax22.tick_params(axis='y', labelcolor='red', labelbottom=False)
    #ax22.tick_params(axis='x', rotation=45)
    ax2.set_xlabel(column)
    ax2.set_xlim(left=-.6)
    ax1.set_title('%s statistics' %column)
    plt.tight_layout()
        

