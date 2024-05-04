import pandas as pd import matplotlib.pyplot as plt 
 
# Load data 
df = pd.read_csv('/content/drive/MyDrive/NETFLX.csv', index_col=
0, parse_dates=True) df.dropna(inplace=True) 
 
# Calculate exponential moving average span = 20  # EMA span (number of periods) ema = df['Close'].ewm(span=span, adjust=False).mean() 
 
# Plot original data and EMA 
plt.plot(df['Close'], label='Original Data') plt.plot(ema, label=f'EMA ({span} periods)') plt.title('Stock Price with Exponential Moving Average') plt.xlabel('Date') plt.ylabel('Price') plt.legend() plt.show() 
