# data to plot
n_groups = 10



freq = (339,431,244,343,54,51,134,189,149,98)
score = (159,185,112,171,27,34,67,89,72,35 )





# create plot
fig, ax = plt.subplots(figsize=(15, 10))
index = np.arange(n_groups)
bar_width = 0.35
opacity = 0.8

rects1 = plt.bar(index, freq, bar_width,
alpha=opacity,
color='b',
label='Frequency')

rects2 = plt.bar(index + bar_width, score, bar_width,
alpha=opacity,
color='g',
label='Accuracy')



plt.xlabel('Patterns')
plt.ylabel('Respective Total')
plt.title('Pattern Frequency-Score')
plt.xticks(index + bar_width, ('Bullish Harami', 'Bearish Harami', 'Bullish Engulfing', 'Bearish Engulfing','Hammer','Hanging Man','Piercing Line','Dark Cloud Cover','Morning Star','Evening Star'))
plt.legend()

plt.tight_layout()
plt.show()

#bu = candlesticks_count.drop(['Bearish_Harami', 'Bearish_Engulfing', 'Green_Hanging', 'Red_Hanging','Dark_Cloud', 'Evening_Star'])
#be = candlesticks_count.drop(['Bullish_Harami', 'Bullish_Engulfing', 'Green_Hammer', 'Red_Hammer','Piercing_Pattern', 'Morning_Star'])
#
#print("Total Bullish:")
#print(bu)
#print("Total Bearish:")
#print(be)
#print("Scores:")
#print(corrected_scores)