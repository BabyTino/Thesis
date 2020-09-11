
# importing necessary modules
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


aapl = pd.read_csv("AMZN.csv")



aapl.isna().sum().sum()




# Inspect the dataframes and check out the data
aapl = aapl.sort_values('date')
aapl = aapl.reset_index(drop=True)
aapl.head()


aapl = aapl.drop(['adjclose'],axis=1)



aapl.shape





def find_trend(data, period: int):
    '''
    Inputs:
    takes in a dataframe and an interger
    Outputs:
    returns True if the trend of the simple moving average over given period is positive, else returns False
    '''
    sma = data['close'].rolling(period).mean() # creates a series with the rolling mean
    diff = sma - sma.shift(1)  # calculates a series of values
    greater_than_0 = diff > 0  # creates a series of bools
    return diff, greater_than_0


aapl['value'], aapl['trend'] = find_trend(aapl, 3)
display(aapl.tail(5))


# In[10]:


aapl['trend'].value_counts()


# In[53]:



# In[63]:





# False means that the trend is down, which makes sense by looking at the closing prices of the stock

# ### Note:
# Since candlestick patterns usually consist of two or more candles, the pattern is not confirmed until the last candle in the pattern forms. For this reason, the index of the candlestick pattern is placed on the index of the last candle in the pattern.
# 
# The shift method basically shifts the entire series down by an integer that is passed into it as an argument. This means that for comparing the price of any day with the previous day, we can shift the series by 1 and compare it with the original series. 
# (Note that the date column is sorted in chronological way)
# 
# ##### Looping through pandas dataframes is incredibly slow and is not recommended. This is why the shift method is used

# In[57]:


def find_bullish_harami(data):
    '''
    Takes in a dataframe containing closing prices of the stock and returns True where bullish harami appears
    '''
    # Opened higher than previous close
    condition_1_BH = data['open'] > data['close'].shift(1) 
    # closed lower than prev open
    condition_2_BH = data['close'] < data['open'].shift(1) 
    # previous candle is red
    condition_3_BH = data['open'].shift(1) > data['close'].shift(1) 
    # the candle is green
    condition_4_BH = data['close'] > data['open'] 
    # must appear in a downtrend
    condition_5_BH = ~ data['trend']
    return condition_1_BH & condition_2_BH & condition_3_BH & condition_4_BH & condition_5_BH


# In[58]:


aapl['Bullish_Harami'] = find_bullish_harami(aapl)


# In[45]:


aapl['Bullish_Harami'].value_counts()


# Looks like we have 220 Bullish Haramis in our dataset. Namely, 2.27% of all the candlesticks.
# The number makes sense, since candlestick patterns don't appear too often. Let's look for bearish haramis now:

# Conditions of a bearish harami are similar to those of bullish harami, only the trend has been reversed:
# - The big candle apprears first and is green
# - The smaller candle appears next and is red 
# 

# In[59]:


def find_bearish_harami(data):
    '''
    
    Takes in a dataframe containing closing prices of the stock and returns True where bearish harami appears
    '''
    Bear_Har_cond_1 = data['close'].shift(1) > data['open']
    Bear_Har_cond_2 = data['close'] > data['open'].shift(1)
    Bear_Har_cond_3 = data['close'].shift(1) > data['open'].shift(1)
    Bear_Har_cond_4 = data['open'] > data['close'] 
    Bear_Har_cond_5 = data['trend']
    
    return Bear_Har_cond_1 & Bear_Har_cond_2 & Bear_Har_cond_3 & Bear_Har_cond_4 & Bear_Har_cond_5


# In[60]:


aapl['Bearish_Harami'] = find_bearish_harami(aapl)


# In[61]:


aapl['Bearish_Harami'].value_counts()


# In[62]:


100 * aapl['Bearish_Harami'].sum()/aapl.shape[0]


# a total of 170 bearish haramis have appeared in the dataset. Namely, 1.75% of the candlesticks are bearish harami

# Next candlestick pattern is the bullish engulfing pattern. A bullish engulfing pattern occurs after a price move lower and indicates higher prices to come. The first candle, in the two-candle pattern, is a down candle. The second candle is a larger up candle, with a real body that fully engulfs the smaller down candle. 

# In[63]:


def find_bullish_engulfing(data):
    '''
    Takes in a dataframe containing closing prices of the stock and returns True where bullish engulfing appears
    '''
    # the second candle must close higher than previous candle
    Bull_Eng_cond_1 = data['close'] > data['open'].shift(1)
    # the second candle must open lower than previous canlde
    Bull_Eng_cond_2 = data['close'].shift(1) > data['open']
    # The second canlde must be green
    Bull_Eng_cond_3 = data['close'] > data['open']
    # The first candle must be red
    Bull_Eng_cond_4 = data['open'].shift(1) > data['close'].shift(1)
    
    Bull_Eng_cond_5 = ~ data['trend']
    
    return Bull_Eng_cond_1 & Bull_Eng_cond_2 & Bull_Eng_cond_3  & Bull_Eng_cond_4 & Bull_Eng_cond_5


# In[64]:


aapl['Bullish_Engulfing'] = find_bullish_engulfing(aapl)


# In[65]:


aapl['Bullish_Engulfing'].value_counts()


# In[66]:


100 * aapl['Bullish_Engulfing'].sum() / aapl.shape[0]


# Next we will look at bearish engulfing pattern, the idea is the same as bullish engulfing, but the colors are reversed.

# In[67]:


def find_bearish_engulfing(data):
    '''
    Takes in a dataframe containing closing prices of the stock and returns True where bearish engulfing appears
    '''
    Bear_Eng_cond_1 = data['open'] > data['close'].shift(1)
    Bear_Eng_cond_2 = data['open'].shift(1) > data['close']
    Bear_Eng_cond_3 = data['close'].shift(1) > data['open'].shift(1)
    Bear_Eng_cond_4 = data['open'] > data['close']
    Bear_Eng_cond_5 = data['trend']
    return Bear_Eng_cond_1 & Bear_Eng_cond_2 & Bear_Eng_cond_3 & Bear_Eng_cond_4 & Bear_Eng_cond_5


# In[68]:


aapl['Bearish_Engulfing'] = find_bearish_engulfing(aapl)


# In[69]:


aapl['Bearish_Engulfing'].value_counts()


# In[70]:


100*aapl['Bearish_Engulfing'].sum()/aapl.shape[0]


# Next up: Hammer: A hammer is a type of bullish reversal candlestick pattern, made up of just one candle, found in price charts of financial assets. The candle looks like a hammer, as it has a long lower wick and a short body at the top of the candlestick with little or no upper wick

# In[71]:


def find_green_hammer(data):
    '''
    Takes in a dataframe containing closing prices of the stock and returns True where green hammer appears
    '''
    # lower shadow at least twice as long as body
    Gr_Ham_cond_1 = (data['open'] - data['low']) > 2*(data['close']-data['open']) 
    # Upper shadow shorter than a tenth of the body
    Gr_Ham_cond_2 = (data['close']-data['open']) > 10*(data['high'] - data['close'])
    # candle should be green
    Gr_Ham_cond_3 = data['close']>data['open']
    # downtrend
    Gr_Ham_cond_4 = ~ data['trend']
    return Gr_Ham_cond_1 & Gr_Ham_cond_2 & Gr_Ham_cond_3 & Gr_Ham_cond_4


# In[72]:


aapl['Green_Hammer'] = find_green_hammer(aapl)


# In[73]:


aapl['Green_Hammer'].value_counts()


# In[74]:


100*aapl['Green_Hammer'].sum()/aapl.shape[0]


# Usually in the candlestick pattern analysis, the color of the hammer is not important. But let's separate them in our study to see if the green hammer has a more bullish reversal power compred to the red one. We will do the color separation for inverted hammer, shooting star and hanging man as well.

# In[75]:


def find_red_hammer(data):
    '''
    Takes in a dataframe containing closing prices of the stock and returns True where red hammer appears'''
    # The wick should be at least twice as long as the body
    Rd_Ham_cond_1 = (data['close'] - data['low']) > 2*(data['open']-data['close'])
    # The lower shadow must be very small, at least 10 times smaller than the body
    Rd_Ham_cond_2 = (data['open']-data['close']) > 10*(data['high'] - data['open']) 
    # candle should be bearish
    Rd_Ham_cond_3 = data['open'] > data['close']
    
    Rd_Ham_cond_4 = ~ data['trend']
    return Rd_Ham_cond_1 & Rd_Ham_cond_2 & Rd_Ham_cond_3 & Rd_Ham_cond_4


# In[76]:


aapl['Red_Hammer'] = find_red_hammer(aapl)


# In[77]:


aapl['Red_Hammer'].value_counts()


# In[78]:


100*aapl['Red_Hammer'].sum()/aapl.shape[0]


# A hanging man is a type of bearish reversal pattern, made up of just one candle, found in an uptrend of price charts of financial assets. It has a long lower wick and a short body at the top of the candlestick with little or no upper wick.
# 

# In[79]:


def find_green_hanging(data):
    '''
    Takes in a dataframe containing closing prices of the stock and returns True where hanging man appears'''
    # lower shadow should be at least twice as long as the body
    Gr_Hang_cond_1 = (data['open']-data['low']) > 2*(data['close']-data['open'])
    # Upper shadow shorter than a tenth of the body
    Gr_Hang_cond_2 = (data['close']-data['open']) > 10*(data['high'] - data['close'])
    # candle should be green
    Gr_Hang_cond_3 = data['close'] > data['open']
    Gr_Hang_cond_4 = data['trend']
    return Gr_Hang_cond_1 & Gr_Hang_cond_2 & Gr_Hang_cond_3 & Gr_Hang_cond_4


# In[80]:


aapl['Green_Hanging'] = find_green_hanging(aapl)


# In[81]:


aapl['Green_Hanging'].value_counts()


# In[82]:


100*aapl['Green_Hanging'].sum()/aapl.shape[0]


# We'll separate the green from red to see their difference in predicting a bearish reversal

# In[83]:


def find_red_hanging(data):
    '''
    Takes in a dataframe containing closing prices of the stock and returns True where hanging man appears'''
    Rd_Hang_cond_1 = (data['close'] - data['low']) > 2*(data['open']-data['close'])
    Rd_Hang_cond_2 = (data['open']-data['close']) > 10*(data['high'] - data['open'])
    Rd_Hang_cond_3 = data['open'] > data['close']
    Rd_Hang_cond_4 = data['trend']
    return Rd_Hang_cond_1 & Rd_Hang_cond_2 & Rd_Hang_cond_3 & Rd_Hang_cond_4


# In[84]:


aapl['Red_Hanging'] = find_red_hanging(aapl)


# In[85]:


aapl['Red_Hanging'].value_counts()


# In[86]:


aapl['Red_Hanging'].sum()/aapl.shape[0] *100


# A piercing pattern is one of a few important candlestick patterns that technical analysts typically spot on a price series chart. This pattern is formed by two consecutive candlestick marks. The first candlestick is red/black signifying a down day and the second is white/green signifying an up day. When a trader is watching for a bullish reversal any red candlestick followed by a white candlestick could be an alert.

# In[87]:


def find_piercing_pattern(data):
    '''
    Takes in a dataframe containing closing prices of the stock and returns True where piercing pattern appears'''
    # Last candle RED
    PP_cond_1 = data['open'].shift(1) > data['close'].shift(1) 
    # This candle GREEN
    PP_cond_2 = data['close'] > data['open'] 
    PP_cond_3 = data['close'].shift(1) > data['open']
    PP_cond_4 = data['close'] > ((data['close'].shift(1) + data['open'].shift(1))/2)
    # NOT ENGULFING 
    PP_cond_5 = data['open'].shift(1) > data['close'] 
    PP_cond_6 = ~data['trend']
    return PP_cond_1 & PP_cond_2 & PP_cond_3 & PP_cond_4 & PP_cond_5 & PP_cond_6


# In[88]:


aapl['Piercing_Pattern'] = find_piercing_pattern(aapl)


# In[89]:


aapl['Piercing_Pattern'].value_counts()


# In[90]:


aapl['Piercing_Pattern'].sum()/aapl.shape[0]*100


# Dark Cloud Cover is a bearish reversal candlestick pattern where a down candle (typically black or red) opens above the close of the prior up candle (typically white or green), and then closes below the midpoint of the up candle. 

# In[91]:


def find_dark_cloud(data):
    '''
    
    Takes in a dataframe containing closing prices of the stock and returns True where dark cloud appears'''
    # Last candle GREEN
    DK_cond_1 = data['close'].shift(1) > data['open'].shift(1) 
    # This candle RED
    DK_cond_2 = data['open'] > data['close'] 
    DK_cond_3 = data['open'] > data['close'].shift(1)
    DK_cond_4 =(data['close'].shift(1) + data['open'].shift(1))/2 > data['close']
    # NOT ENGULFING
    DK_cond_5 = data['close'] > data['open'].shift(1) 
    DK_cond_6 = data['trend']
    return DK_cond_1 & DK_cond_2 & DK_cond_3 & DK_cond_4 & DK_cond_5 & DK_cond_6


# In[92]:


aapl['Dark_Cloud'] = find_dark_cloud(aapl)


# In[93]:


aapl['Dark_Cloud'].value_counts()


# In[94]:


aapl['Dark_Cloud'].sum()/aapl.shape[0]*100


# The Morning Star is a pattern seen in a candlestick chart, a type of chart used by stock analysts to describe and predict price movements of a security, derivative, or currency over time

# In[95]:


def find_morning_star(data):
    '''
    Takes in a dataframe containing closing prices of the stock and returns True where morning star appears'''
    # First candle RED
    MS_cond_1 = data['open'].shift(2) > data['close'].shift(2) 
    # Third candle Green
    MS_cond_2 = data['close'] > data['open']
    # Third candle closes higher than the middle one
    MS_cond_3 = (data['close'] > data['close'].shift(1)) 
    
    MS_cond_4 = data['close'] > (data['open']+data['close'])/2
    MS_cond_5 = data['close'].shift(1) < data['open']
    MS_cond_6 = data['open'].shift(1) < data['open']
    MS_cond_7 = (data['close'].shift(1) < data['close'].shift(2)) & (data['open'].shift(1) < data['close'].shift(2))
    MS_cond_8 = ~ data['trend']
    return MS_cond_1 & MS_cond_2 & MS_cond_3 & MS_cond_4 & MS_cond_5 & MS_cond_6 & MS_cond_7 & MS_cond_8
    


# In[96]:


aapl['Morning_Star'] = find_morning_star(aapl)


# In[97]:


aapl['Morning_Star'].value_counts()


# In[98]:


aapl['Morning_Star'].sum()/aapl.shape[0]*100


# An evening star is a bearish candlestick pattern consisting of three candles: a large white candlestick, a small-bodied candle, and a red candle.

# In[99]:


def find_evening_star(data):
    '''
    Takes in a dataframe containing closing prices of the stock and returns True where evening star appears'''
    # First candle GREEN
    ES_cond_1 = data['close'].shift(2) > data['open'].shift(2)
    ES_cond_2 = data['open'] > data['close'] #nexy candle RED
    ES_cond_3 = data['close'] < (data['close'].shift(2) + data['open'].shift(2))/2
    ES_cond_4 = data['open'].shift(1) > data['close'].shift(2)
    ES_cond_5 = data['close'].shift(1) > data['close'].shift(2)
    ES_cond_6 = data['open'].shift(1) > data['open']
    ES_cond_7 = data['close'].shift(1) > data['open'] 
    ES_cond_8 = data['trend']
    return ES_cond_1 & ES_cond_2 & ES_cond_3 & ES_cond_4 & ES_cond_5 & ES_cond_6 & ES_cond_7 & ES_cond_8


# In[100]:


aapl['Evening_Star'] = find_evening_star(aapl)


# In[101]:


aapl['Evening_Star'].value_counts()


# In[102]:


aapl['Evening_Star'].sum()/aapl.shape[0]*100


# A kicker pattern is a two-bar candlestick pattern that is used to predict a change in the direction of the trend for an asset's price. This pattern is characterized by a very sharp reversal in price over the span of two candlesticks; traders use it to determine which group of market participants is in control of the direction. The pattern points to a strong change in investors' attitude surrounding a security. This usually occurs following the release of valuable information about a company, industry or an economy.


# Now that we have identified where our candlestick patterns of interest appear, let's crunch some numbers together
# 
# 

# In[168]:


candlesticks_count = aapl.drop(['open','close','high','low','volume','date','value','trend],axis=1).sum()


# In[169]:


candlesticks_count


# In[129]:


get_ipython().run_line_magic('matplotlib', 'inline')
ax = candlesticks_count.plot.bar()
ax.grid()
ax.set_xlabel('Pattern')
ax.set_ylabel('Count')
ax.set_title('Frequency of Patterns in Amazon Stock')


# This shows how some candlestick patterns can appear frequently but others are quite rare to find!

# In[128]:


aapl.drop(['open','close','high','low','volume','date','value','trend'],axis=1).sum().sum()


# A total of 994 candlestick patterns have occured.

# ##### Prediction score:
# let's define the prediction score to be the fraction of the times that the pattern correctly predicted a price change to the total number of times the pattern appeared. 

# In[130]:


def get_prediction_score(data, candlestick_pattern: str):
    '''
    takes in a dataframe and the name of a candlestick pattern and calculates the fraction of times the candlestick patter
    managed to predict the market correctly
    '''
    initial_prices = data['close'][data[candlestick_pattern]==1]
    next_price_point = data['close'][data[candlestick_pattern].shift(1)==1]
    price_increased = next_price_point.reset_index(drop=True) > initial_prices.reset_index(drop=True)
    price_dropped = next_price_point.reset_index(drop=True) < initial_prices.reset_index(drop=True)
    if candlestick_pattern in ['Bullish_Harami', 'Bullish_Engulfing', 'Green_Hammer', 'Red_Hammer', 'Piercing_Pattern',
                               'Morning_Star', 'Bull_Kicker', 'Green_Inverted_Hammer', 'Red_Inverted_Hammer']:
        prediction_score = price_increased.sum()/len(price_increased)
        return prediction_score
    # elif used rather than else to prevent typos to misclassify bullish and bearish signals
    elif candlestick_pattern in ['Bearish_Harami', 'Bearish_Engulfing', 'Green_Hanging', 'Red_Hanging', 'Dark_Cloud',
                                 'Evening_Star', 'Bear_Kicker', 'Green_Shooting_Star', 'Red_Shooting_Star']:
        prediction_score = price_dropped.sum()/len(price_dropped)
        return prediction_score
    else:
        print(f'Sorry, {candlestick_pattern} was not found in our list of modeled candlestick pattern ')
        


# In[131]:


scores = pd.Series()
candlesticks = ['Bullish_Harami', 'Bearish_Harami','Bullish_Engulfing', 'Bearish_Engulfing','Green_Hammer', 'Red_Hammer',
                    'Green_Hanging', 'Red_Hanging','Piercing_Pattern', 'Dark_Cloud', 'Morning_Star','Evening_Star',
                    'Bull_Kicker', 'Bear_Kicker', 'Green_Shooting_Star', 'Red_Shooting_Star',
                    'Green_Inverted_Hammer', 'Red_Inverted_Hammer' ]
for pattern in candlesticks:
    scores[pattern] = get_prediction_score(aapl, pattern)


# ## Evaluating The Results
# Now it's time to look at the prediction score of each pattern and see if we can rely on them for trading!

# In[132]:


ax = scores.plot.bar(color='green')
ax.set_title('Prediction Score of Candlestick Patterns')
ax.set_ylim(0,1)
ax.grid()


# The higher the prediction score, the more reliable the candlestick pattern is. 

# ## Important Note:
# You might think, there is a big bias in the scores that are plotted above. Since the price of the Apple stock has had an uptrend since the initial public offering, there should have been more bullish days than bearish days. Thus if you choose a random day in the history of the stock, it is more likely that the price will be higher the next day. In other words, the bullish signals have an unfair advantage in our scoring system. That is a fair assumption. Let's check this together

# In[133]:


def get_bullish_fraction(data):
    '''takes in a dataframe and returns the ratio of days where close price has increased'''
    bullish_days = data['close'] > data['close'].shift(1)
    return bullish_days.sum()/len(data)


# In[134]:


bullish_fraction = get_bullish_fraction(aapl)
plt.pie([bullish_fraction, 1-bullish_fraction], labels=['Bullish_Days','Bearish_Days'])
print('Bullish_day_fraction', np.round(bullish_fraction,3))


# Well I guess that is not what you expected. The number of bullish and bearish days are close to the number of bearish days. One explanation for this could be that the magnitude of price gains were higher than the magnitude of drops on average, thus the stock has gained over time despite having fewere bullish days. Don't want to go off-topic, but let's check this out quickly.

# In[135]:


where_loss_happened = aapl['close'] > aapl['close'].shift(-1)
price_losses = aapl['close'][where_loss_happened].reset_index(drop=True) - aapl['close'][where_loss_happened.shift(1).fillna(False)].reset_index(drop=True)
abs(np.mean(price_losses))


# In[136]:


where_gain_happened = aapl['close'] < aapl['close'].shift(-1)
price_gains = aapl['close'][where_gain_happened].reset_index(drop=True) - aapl['close'][where_gain_happened.shift(1).fillna(False)].reset_index(drop=True)
abs(np.mean(price_gains))


# As you can see, the average daily gain is higher than average daily loss. 

# Now let's write a function to correct the prediction scores of each candlestick patterns

# In[137]:


def correct_prediction_score(data, candlestick_pattern: str):
    '''
    Takes in a dataframe and the name of a candlestick pattern and returns a new accuracy score which is corrected
    based on the ratio of bullish to bearish days
    '''
    bullish_fraction = get_bullish_fraction(data)
    initial_prices = data['close'][data[candlestick_pattern]==1]
    next_price_point = data['close'][data[candlestick_pattern].shift(1)==1]
    price_increased = next_price_point.reset_index(drop=True) > initial_prices.reset_index(drop=True)
    price_dropped = next_price_point.reset_index(drop=True) < initial_prices.reset_index(drop=True)
    if candlestick_pattern in ['Bullish_Harami', 'Bullish_Engulfing', 'Green_Hammer', 'Red_Hammer', 'Piercing_Pattern',
                               'Morning_Star', 'Bull_Kicker', 'Green_Inverted_Hammer', 'Red_Inverted_Hammer']:
        prediction_score = price_increased.sum()/len(price_increased)
        prediction_score *= 0.5/bullish_fraction
        return prediction_score
    # elif used rather than else to prevent typos to misclassify bullish and bearish signals
    elif candlestick_pattern in ['Bearish_Harami', 'Bearish_Engulfing', 'Green_Hanging', 'Red_Hanging', 'Dark_Cloud',
                                 'Evening_Star', 'Bear_Kicker', 'Green_Shooting_Star', 'Red_Shooting_Star']:
        prediction_score = price_dropped.sum()/len(price_dropped)
        prediction_score *= 0.5/(1-bullish_fraction)
        return prediction_score
    else:
        print(f'Sorry, {candlestick_pattern} was not found in our list of modeled candlestick pattern ')
        


# In[138]:


corrected_scores = pd.Series()
for pattern in candlesticks:
    corrected_scores[pattern] = correct_prediction_score(aapl, pattern)
    


# In[139]:


ax = corrected_scores.plot.bar()
ax.set_ylim(0,1)
ax.grid()
ax.set_title('Score of patterns in Apple stock')


# #### Analysis
# First we need to understand the scoring system. A score of 1 is the perfect score and it means that whenever the pattern has appeared, the market trend has changed in the way that the pattern predicts, e.g. if bullish kicker had a score of 1, it would mean that every time the pattern had been observed, the next candle was bullish. 
# 
# On the other hand, a score of 0 means the worst possible score, and it means that the candle has made the wrong prediction every single time. 
# 
# The middle ground is the score of 0.5, which is the score that you would get if you were to randomly guess the trend of the market. Basically this means you are right half of the time. 
# 
# Looking at the data, we observe that most of the candles seem to have a score of roughly 0.5, which means that they do not offer any predictive power whatsoever. 
# In the example of Bull kicker and Bear kicker patterns, the number of observations is too low to make a judgement. Maybe it would help if we repeated the same analysis on more stock data to increase the number of our observations and make a better judgement. Let's do it!

# In[151]:


def analyze_candlestick_accuracy(data, stock_name:str):
    '''
    Does all the above mentioned analysis and plots the frequency and corrected score of the patterns
    '''
    data = data.sort_values('date')
    data = data.reset_index(drop=True)
    data = data.drop(['adjclose'],axis=1)
    data['value'], data['trend'] = find_trend(data, 3)
    data['Bullish_Harami'] = find_bullish_harami(data)
    data['Bearish_Harami'] = find_bearish_harami(data)
    data['Bullish_Engulfing'] = find_bullish_engulfing(data)
    data['Bearish_Engulfing'] = find_bearish_engulfing(data)
    data['Green_Hammer'] = find_green_hammer(data)
    data['Red_Hammer'] = find_red_hammer(data)
    data['Green_Hanging'] = find_green_hanging(data)
    data['Red_Hanging'] = find_red_hanging(data)
    data['Piercing_Pattern'] = find_piercing_pattern(data)
    data['Dark_Cloud'] = find_dark_cloud(data)
    data['Morning_Star'] = find_morning_star(data)
    data['Evening_Star'] = find_evening_star(data)
    data['Bull_Kicker'] = find_bull_kicker(data)
    data['Bear_Kicker'] = find_bear_kicker(data)
    data['Green_Shooting_Star'] = find_green_shooting_star(data)
    data['Red_Shooting_Star'] = find_red_shooting_star(data)
    data['Green_Inverted_Hammer'] = find_green_inverted_hammer(data)
    data['Red_Inverted_Hammer'] = find_red_inverted_hammer(data)
    
    candlesticks_count = data.drop(['open','close','high','low','volume','date','value','trend'],axis=1).sum()
    ax1 = candlesticks_count.plot.bar()
    ax1.grid()
    ax1.set_xlabel('Pattern')
    ax1.set_ylabel('Count')
    ax1.set_title(stock_name)
    candlesticks = ['Bullish_Harami', 'Bearish_Harami','Bullish_Engulfing', 'Bearish_Engulfing','Green_Hammer', 'Red_Hammer',
                    'Green_Hanging', 'Red_Hanging','Piercing_Pattern', 'Dark_Cloud', 'Morning_Star','Evening_Star',
                    'Bull_Kicker', 'Bear_Kicker', 'Green_Shooting_Star', 'Red_Shooting_Star',
                    'Green_Inverted_Hammer', 'Red_Inverted_Hammer' ]
    corrected_scores = pd.Series()
    for pattern in candlesticks:
        corrected_scores[pattern] = correct_prediction_score(data, pattern)
    return corrected_scores


# In[152]:


def plot_scores(corrected_scores, stock_name:str):
    ax = corrected_scores.plot.bar(color='purple')
    ax.set_ylim(0,1)
    ax.set_title(stock_name)
    ax.grid()
    


# In[153]:


Amazon = pd.read_csv('AMZN.csv')


# In[154]:


corrected_scores_amzn = analyze_candlestick_accuracy(Amazon, 'Frequency of Patterns in Amazon Stock')


# In[155]:


plot_scores(corrected_scores_amzn, 'Pattern Scores in Amazon Stock')


# This is absolutely interesting! Look at the count of each pattern. The patterns that appear more often seem to score closer to 0.5, which is the equivalent of randomly trading. Let's do the process for other stocks!

# In[ ]:


nflx = pd.read_csv('NFLX.csv')


# In[ ]:


corrected_scores_nflx = analyze_candlestick_accuracy(nflx, 'Frequency of Patterns in Netflix Stock')


# In[ ]:


plot_scores(corrected_scores_nflx, 'Score of Patterns in Netflix Stock')


# #### The correlation is observed again, if the number of observations increases, the score approaches 0.5, which is the equivalent of randomly trading

# This is a clear proof that candlestick patterns by themselves cannot be reliable indicators for future market trends. While there is value in looking at candlestick patterns to understand the market better, they should not be treated as trustworthy indicators of market movement. 

# 
# disclaimer: This post by no means is providing financial advice to anyone, and is merely a statistical analysis of how well candlestick patterns have been able to predict the market

# In[ ]:




