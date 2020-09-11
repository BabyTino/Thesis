import yfinance as yf

Facebook = yf.Ticker("FB")
Amazon = yf.Ticker("AMZN")
Apple = yf.Ticker("AAPL")
Netflix = yf.Ticker("NFLX")
Google = yf.Ticker("GOOG")


# See historical market data
fb = Facebook.history(period="max")
amzn = Amazon.history(period="max")
aapl = Apple.history(period="max")
nflx = Netflix.history(period="max")
goog = Google.history(period="max")

## Downloadd Historical market data 
#data_df = yf.download("AMZN", start="1980-12-12", end="2019-04-18")
#data_df.to_csv('AMZN.csv')

