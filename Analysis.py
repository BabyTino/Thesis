# importing necessary modules
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import mplfinance as mpf


fb = pd.read_csv("AAPL.csv")

# Check for missing data
fb.isna().sum().sum()

# Inspect the dataframes and check out the data
fb = fb.sort_values('Date')
fb = fb.reset_index(drop=True)
fb.head()



# Drop uneeded Columns
fb = fb.drop(['Adj Close'],axis=1)
fb.head()
fb.shape

# Defining a technical Indicator - Simple Moving Average
def find_Trend(data, period: int):
    '''
    Inputs:
    takes in a dataframe and an interger
    Outputs:
    returns True if the Trend of the simple moving average over given period is positive, else returns False
    '''
    sma = data['Close'].rolling(period).mean() # creates a series with the rolling mean
    diff = sma - sma.shift(6)  # calculates a series of values
    greater_than_0 = diff > 0  # creates a series of bools
    return diff, greater_than_0

# False inddicates a Downwards Trend and Vice-versa
fb['value'], fb['Trend'] = find_Trend(fb, 3)
print(fb.tail(5))
print(fb['Trend'].value_counts())


# Defining a pattern
    
def find_bullish_harami(data):
    '''
    Takes in a dataframe containing closing prices of the stock and returns True where bullish harami appears
    '''
    # Opened Higher than previous Close
    condition_1_BH = data['Open'] > data['Close'].shift(1) 
    # Closed Lower than prev Open
    condition_2_BH = data['Close'] < data['Open'].shift(1) 
    # previous candle is red
    condition_3_BH = data['Open'].shift(1) > data['Close'].shift(1) 
    # the candle is green
    condition_4_BH = data['Close'] > data['Open'] 
    # must appear in a downTrend
    condition_5_BH = ~ data['Trend']
    return condition_1_BH & condition_2_BH & condition_3_BH & condition_4_BH & condition_5_BH


def find_bearish_harami(data):
    '''
    
    Takes in a dataframe containing closing prices of the stock and returns True where bearish harami appears
    '''
    Bear_Har_cond_1 = data['Close'].shift(1) > data['Open']
    Bear_Har_cond_2 = data['Close'] > data['Open'].shift(1)
    Bear_Har_cond_3 = data['Close'].shift(1) > data['Open'].shift(1)
    Bear_Har_cond_4 = data['Open'] > data['Close'] 
    Bear_Har_cond_5 = data['Trend']
    
    return Bear_Har_cond_1 & Bear_Har_cond_2 & Bear_Har_cond_3 & Bear_Har_cond_4 & Bear_Har_cond_5


def find_bullish_engulfing(data):
    '''
    Takes in a dataframe containing closing prices of the stock and returns True where bullish engulfing appears
    '''
    # the second candle must Close Higher than previous candle
    Bull_Eng_cond_1 = data['Close'] > data['Open'].shift(1)
    # the second candle must Open Lower than previous canlde
    Bull_Eng_cond_2 = data['Close'].shift(1) > data['Open']
    # The second canlde must be green
    Bull_Eng_cond_3 = data['Close'] > data['Open']
    # The first candle must be red
    Bull_Eng_cond_4 = data['Open'].shift(1) > data['Close'].shift(1)
    
    Bull_Eng_cond_5 = ~ data['Trend']
    
    return Bull_Eng_cond_1 & Bull_Eng_cond_2 & Bull_Eng_cond_3  & Bull_Eng_cond_4 & Bull_Eng_cond_5


def find_bearish_engulfing(data):
    '''
    Takes in a dataframe containing closing prices of the stock and returns True where bearish engulfing appears
    '''
    Bear_Eng_cond_1 = data['Open'] > data['Close'].shift(1)
    Bear_Eng_cond_2 = data['Open'].shift(1) > data['Close']
    Bear_Eng_cond_3 = data['Close'].shift(1) > data['Open'].shift(1)
    Bear_Eng_cond_4 = data['Open'] > data['Close']
    Bear_Eng_cond_5 = data['Trend']
    return Bear_Eng_cond_1 & Bear_Eng_cond_2 & Bear_Eng_cond_3 & Bear_Eng_cond_4 & Bear_Eng_cond_5

def find_white_hammer(data):
    '''
    Takes in a dataframe containing closing prices of the stock and returns True where green hammer appears
    '''
    # Lower shadow at least twice as long as body
    Gr_Ham_cond_1 = (data['Open'] - data['Low']) > 2*(data['Close']-data['Open']) 
    # Upper shadow shorter than a tenth of the body
    Gr_Ham_cond_2 = (data['Close']-data['Open']) > 10*(data['High'] - data['Close'])
    # candle should be green
    Gr_Ham_cond_3 = data['Close']>data['Open']
    # downTrend
    Gr_Ham_cond_4 = ~ data['Trend']
    return Gr_Ham_cond_1 & Gr_Ham_cond_2 & Gr_Ham_cond_3 & Gr_Ham_cond_4

def find_black_Hammer(data):
    '''
    Takes in a dataframe containing closing prices of the stock and returns True where red hammer appears'''
    # The wick should be at least twice as long as the body
    Rd_Ham_cond_1 = (data['Close'] - data['Low']) > 2*(data['Open']-data['Close'])
    # The Lower shadow must be very small, at least 10 times smaller than the body
    Rd_Ham_cond_2 = (data['Open']-data['Close']) > 10*(data['High'] - data['Open']) 
    # candle should be bearish
    Rd_Ham_cond_3 = data['Open'] > data['Close']
    
    Rd_Ham_cond_4 = ~ data['Trend']
    return Rd_Ham_cond_1 & Rd_Ham_cond_2 & Rd_Ham_cond_3 & Rd_Ham_cond_4

def find_white_hanging(data):
    '''
    Takes in a dataframe containing closing prices of the stock and returns True where hanging man appears'''
    # Lower shadow should be at least twice as long as the body
    Gr_Hang_cond_1 = (data['Open']-data['Low']) > 2*(data['Close']-data['Open'])
    # Upper shadow shorter than a tenth of the body
    Gr_Hang_cond_2 = (data['Close']-data['Open']) > 10*(data['High'] - data['Close'])
    # candle should be green
    Gr_Hang_cond_3 = data['Close'] > data['Open']
    Gr_Hang_cond_4 = data['Trend']
    return Gr_Hang_cond_1 & Gr_Hang_cond_2 & Gr_Hang_cond_3 & Gr_Hang_cond_4

def find_black_hanging(data):
    '''
    Takes in a dataframe containing closing prices of the stock and returns True where hanging man appears'''
    Rd_Hang_cond_1 = (data['Close'] - data['Low']) > 2*(data['Open']-data['Close'])
    Rd_Hang_cond_2 = (data['Open']-data['Close']) > 10*(data['High'] - data['Open'])
    Rd_Hang_cond_3 = data['Open'] > data['Close']
    Rd_Hang_cond_4 = data['Trend']
    return Rd_Hang_cond_1 & Rd_Hang_cond_2 & Rd_Hang_cond_3 & Rd_Hang_cond_4

def find_piercing_pattern(data):
    '''
    Takes in a dataframe containing closing prices of the stock and returns True where piercing pattern appears'''
    # Last candle RED
    PP_cond_1 = data['Open'].shift(1) > data['Close'].shift(1) 
    # This candle GREEN
    PP_cond_2 = data['Close'] > data['Open'] 
    PP_cond_3 = data['Close'].shift(1) > data['Open']
    PP_cond_4 = data['Close'] > ((data['Close'].shift(1) + data['Open'].shift(1))/2)
    # NOT ENGULFING 
    PP_cond_5 = data['Open'].shift(1) > data['Close'] 
    PP_cond_6 = ~data['Trend']
    return PP_cond_1 & PP_cond_2 & PP_cond_3 & PP_cond_4 & PP_cond_5 & PP_cond_6

def find_dark_cloud(data):
    '''
    
    Takes in a dataframe containing closing prices of the stock and returns True where dark cloud appears'''
    # Last candle GREEN
    DK_cond_1 = data['Close'].shift(1) > data['Open'].shift(1) 
    # This candle RED
    DK_cond_2 = data['Open'] > data['Close'] 
    DK_cond_3 = data['Open'] > data['Close'].shift(1)
    DK_cond_4 =(data['Close'].shift(1) + data['Open'].shift(1))/2 > data['Close']
    # NOT ENGULFING
    DK_cond_5 = data['Close'] > data['Open'].shift(1) 
    DK_cond_6 = data['Trend']
    return DK_cond_1 & DK_cond_2 & DK_cond_3 & DK_cond_4 & DK_cond_5 & DK_cond_6

def find_morning_star(data):
    '''
    Takes in a dataframe containing closing prices of the stock and returns True where morning star appears'''
    # First candle RED
    MS_cond_1 = data['Open'].shift(2) > data['Close'].shift(2) 
    # Third candle Green
    MS_cond_2 = data['Close'] > data['Open']
    # Third candle Closes Higher than the middle one
    MS_cond_3 = (data['Close'] > data['Close'].shift(1)) 
    
    MS_cond_4 = data['Close'] > (data['Open']+data['Close'])/2
    MS_cond_5 = data['Close'].shift(1) < data['Open']
    MS_cond_6 = data['Open'].shift(1) < data['Open']
    MS_cond_7 = (data['Close'].shift(1) < data['Close'].shift(2)) & (data['Open'].shift(1) < data['Close'].shift(2))
    MS_cond_8 = ~ data['Trend']
    return MS_cond_1 & MS_cond_2 & MS_cond_3 & MS_cond_4 & MS_cond_5 & MS_cond_6 & MS_cond_7 & MS_cond_8

def find_evening_star(data):
    '''
    Takes in a dataframe containing closing prices of the stock and returns True where evening star appears'''
    # First candle GREEN
    ES_cond_1 = data['Close'].shift(2) > data['Open'].shift(2)
    ES_cond_2 = data['Open'] > data['Close'] #nexy candle RED
    ES_cond_3 = data['Close'] < (data['Close'].shift(2) + data['Open'].shift(2))/2
    ES_cond_4 = data['Open'].shift(1) > data['Close'].shift(2)
    ES_cond_5 = data['Close'].shift(1) > data['Close'].shift(2)
    ES_cond_6 = data['Open'].shift(1) > data['Open']
    ES_cond_7 = data['Close'].shift(1) > data['Open'] 
    ES_cond_8 = data['Trend']
    return ES_cond_1 & ES_cond_2 & ES_cond_3 & ES_cond_4 & ES_cond_5 & ES_cond_6 & ES_cond_7 & ES_cond_8

# Indentify Candlestick Patterns
fb['Bullish_Harami'] = find_bullish_harami(fb)  
fb['Bearish_Harami'] = find_bearish_harami(fb)
fb['Bullish_Engulfing'] = find_bullish_engulfing(fb)
fb['Bearish_Engulfing'] = find_bearish_engulfing(fb)
fb['White_Hammer'] = find_white_hammer(fb)
fb['Black_Hammer'] = find_black_Hammer(fb)
fb['White_Hanging'] = find_white_hanging(fb)
fb['Black_Hanging'] = find_black_hanging(fb)
fb['Piercing_Pattern'] = find_piercing_pattern(fb)
fb['Dark_Cloud'] = find_dark_cloud(fb)    
fb['Morning_Star'] = find_morning_star(fb)
fb['Evening_Star'] = find_evening_star(fb)


# Evaluate Candlestick Pattern Occurence 

candlesticks_count = fb.drop(['Open','Close','High','Low','Volume','Date','value','Trend'],axis=1).sum()

freq = candlesticks_count.plot.bar(color='cornflowerblue')
freq.grid()
freq.set_title('Frequency of Patterns in Apple Stock')
plt.show()


print(candlesticks_count)

print("Percentage of Candlestick patterns identified amongst",fb.shape[0],"candles:", candlesticks_count.sum()/fb.shape[0]*100)

# Define Prediction Score  
# Checks for accuracy of reversal pattern indications

def get_prediction_score(data, candlestick_pattern: str):
    '''
    takes in a dataframe and the name of a candlestick pattern and calculates the fraction of times the candlestick patter
    managed to predict the market correctly
    '''
    initial_prices = data['Close'][data[candlestick_pattern]==1]
    next_price_point = data['Close'][data[candlestick_pattern].shift(1)==1]
    price_increased = next_price_point.reset_index(drop=True) > initial_prices.reset_index(drop=True)
    price_dropped = next_price_point.reset_index(drop=True) < initial_prices.reset_index(drop=True)
    if candlestick_pattern in ['Bullish_Harami', 'Bullish_Engulfing', 'White_Hammer', 'Black_Hammer', 'Piercing_Pattern',
                               'Morning_Star']:
        prediction_score = price_increased.sum()/len(price_increased)
        return prediction_score
    # elif used rather than else to prevent typos to misclassify bullish and bearish signals
    elif candlestick_pattern in ['Bearish_Harami', 'Bearish_Engulfing', 'White_Hanging', 'Black_Hanging', 'Dark_Cloud',
                                 'Evening_Star']:
        prediction_score = price_dropped.sum()/len(price_dropped)
        return prediction_score
    else:
        print(f'Sorry, {candlestick_pattern} was not found in our list of modeled candlestick pattern ')


results = pd.Series()
candlesticks = ['Bullish_Harami', 'Bearish_Harami','Bullish_Engulfing', 'Bearish_Engulfing','White_Hammer', 'Black_Hammer',
                'White_Hanging', 'Black_Hanging','Piercing_Pattern', 'Dark_Cloud', 'Morning_Star','Evening_Star']

for pattern in candlesticks:
    results[pattern] = get_prediction_score(fb, pattern)
    
 #Observing results
 #1 is the highest accuracy score possible.

ax = results.plot.bar(color='mediumseagreen')
ax.set_title('Prediction Score of Candlestick Patterns')
ax.set_ylim(0,1)
ax.grid()
plt.show()


 #Correct potential Bias in Data  - Chapter 5
 #As Apple has been in a general Upwards trend since the start, there should be more Bullish days


def get_bullish_fraction(data):
    '''takes in a dataframe and returns the ratio of days where Close price has increased'''
    bullish_days = data['Close'] > data['Close'].shift(1)
    return bullish_days.sum()/len(data)

bullish_fraction = get_bullish_fraction(fb)
#plt.pie([bullish_fraction, 1-bullish_fraction], autopct='%1.1f%%', labels=['Bullish_Days','Bearish_Days'])
#print('Fraction of Bullish days: ', np.round(bullish_fraction,3))
#plt.show()

locate_loss = fb['Close'] > fb['Close'].shift(-1)
price_losses = fb['Close'][locate_loss].reset_index(drop=True) - fb['Close'][locate_loss.shift(1).fillna(False)].reset_index(drop=True)
print(abs(np.mean(price_losses)))

locate_gain = fb['Close'] < fb['Close'].shift(-1)
price_gains = fb['Close'][locate_gain].reset_index(drop=True) - fb['Close'][locate_gain.shift(1).fillna(False)].reset_index(drop=True)
abs(np.mean(price_gains))



# Rectify Bias in data

def correct_prediction_score(data, candlestick_pattern: str):
    '''
    Takes in a dataframe and the name of a candlestick pattern and returns a new accuracy score which is corrected
    based on the ratio of bullish to bearish days
    '''
    bullish_fraction = get_bullish_fraction(data)
    initial_prices = data['Close'][data[candlestick_pattern]==1]
    next_price_point = data['Close'][data[candlestick_pattern].shift(1)==1]
    price_increased = next_price_point.reset_index(drop=True) > initial_prices.reset_index(drop=True)
    price_dropped = next_price_point.reset_index(drop=True) < initial_prices.reset_index(drop=True)
    if candlestick_pattern in ['Bullish_Harami', 'Bullish_Engulfing', 'White_Hammer', 'Black_Hammer', 'Piercing_Pattern',
                               'Morning_Star']:
        prediction_score = price_increased.sum()/len(price_increased)
        prediction_score *= 0.5/bullish_fraction
        return prediction_score
    # elif used rather than else to prevent typos to misclassify bullish and bearish signals
    elif candlestick_pattern in ['Bearish_Harami', 'Bearish_Engulfing', 'White_Hanging', 'Black_Hanging', 'Dark_Cloud',
                                 'Evening_Star']:
        prediction_score = price_dropped.sum()/len(price_dropped)
        prediction_score *= 0.5/(1-bullish_fraction)
        return prediction_score
    else:
        print(f'Sorry, {candlestick_pattern} was not found in our list of modeled candlestick pattern ')

corrected_scores = pd.Series()
for pattern in candlesticks:
    corrected_scores[pattern] = correct_prediction_score(fb, pattern)
ax = corrected_scores.plot.bar(color='forestgreen')    
ax.set_ylim(0,1)
ax.grid()
ax.set_title('Prediction Score of patterns in Netflix stock')
plt.show()