import pandas as pd
import holidays
import talib


# Constants
SMA5_PERIOD = 5  # Period for 5-period Simple Moving Average
ROLLING_WINDOW = 20  # Period for the rolling window used for statistics like Rolling Mean and Rolling Standard Deviation
RSI_PERIOD = 14  # Period for Relative Strength Index (RSI) calculation
EMA_PERIOD = 12  # Period for Exponential Moving Average (EMA) calculation
MACD_FASTPERIOD = 12  # Fast period for Moving Average Convergence Divergence (MACD) calculation
MACD_SLOWPERIOD = 26  # Slow period for MACD calculation
MACD_SIGNALPERIOD = 9  # Signal period for MACD calculation
SMA10_PERIOD = 10  # Period for 10-period Simple Moving Average
SMA20_PERIOD = 20  # Period for 20-period Simple Moving Average
SMA50_PERIOD = 50  # Period for 50-period Simple Moving Average
BBANDS_PERIOD = 20  # Period for Bollinger Bands calculation
ROC_PERIOD = 12  # Period for Rate of Change (ROC) calculation
ATR_PERIOD = 14  # Period for Average True Range (ATR) calculation
CCI_PERIOD = 20  # Period for Commodity Channel Index (CCI) calculation
WILLR_PERIOD = 14  # Period for Williams %R (WILLR) calculation
STOCH_FASTK_PERIOD = 5  # Fast period for Stochastic Oscillator calculation
STOCH_SLOWK_PERIOD = 3  # SlowK period for Stochastic Oscillator calculation
STOCH_SLOWD_PERIOD = 3  # SlowD period for Stochastic Oscillator calculation
MFI_PERIOD = 14  # Period for Money Flow Index (MFI) calculation
us_holidays = holidays.UnitedStates()


def process_indicators(df):
    df = df.copy()
    # Convert 'Timestamp' to datetime and sort
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
    df.sort_values(by='timestamp', inplace=True)
    
    # Calculate day of the week
    df['day_of_week'] = df['timestamp'].dt.dayofweek
    
    # Check for holidays
    df['is_holiday'] = df['timestamp'].dt.date.apply(
        lambda x: int(x in us_holidays))
    
    # Compute Unix timestamp
    df['timestamp'] = df['timestamp'].astype('int64') // 10**9
    
    # Calculate Returns
    df['returns'] = (df['close'] / df['close'].shift(1) - 1)
    
    # Calculate general statistics
    df['avg_Price'] = df["close"].mean()
    df['avg_Returns'] = df['returns'].mean()
    df['volatility'] = df['returns'].std()
    df['volume_Volatility'] = df['volume'].std()

    # Calculate Moving Average, Price and Volume change
    df['moving_Avg'] = df['close'].rolling(window=SMA5_PERIOD).mean()
    df['price_Change'] = df['close'].pct_change() * 100
    df['volume_Change'] = df['volume'].diff()
    
    # Calculate rolling window statistics
    df['Rolling_Mean'] = df['close'].rolling(window=ROLLING_WINDOW).mean()
    df['Rolling_Std'] = df['close'].rolling(window=ROLLING_WINDOW).std()

    # Technical Indicators
    df['RSI'] = talib.RSI(df['close'], timeperiod=RSI_PERIOD)
    df['EMA'] = talib.EMA(df['close'], timeperiod=EMA_PERIOD)
    df['MACD'] = talib.MACD(df['close'], fastperiod=MACD_FASTPERIOD,
                            slowperiod=MACD_SLOWPERIOD,
                            signalperiod=MACD_SIGNALPERIOD)[0]
    df['SMA_5'] = talib.SMA(df['close'], timeperiod=SMA5_PERIOD)
    df['SMA_10'] = talib.SMA(df['close'], timeperiod=SMA10_PERIOD)
    df['SMA_20'] = talib.SMA(df['close'], timeperiod=SMA20_PERIOD)
    df['SMA_50'] = talib.SMA(df['close'], timeperiod=SMA50_PERIOD)
    upper, middle, lower = talib.BBANDS(
        df['close'], timeperiod=BBANDS_PERIOD
    )
    df['BBANDS_Upper'] = upper
    df['BBANDS_Middle'] = middle
    df['BBANDS_Lower'] = lower
    df['VWAP'] = (
        df['volume'] * (df['high'] + df['low'] + df['close']) / 3
    ).cumsum() / df['volume'].cumsum()
    df['ROC'] = talib.ROC(df['close'], timeperiod=ROC_PERIOD)
    df['ATR'] = talib.ATR(
        df['high'], df['low'], df['close'], timeperiod=ATR_PERIOD
    )
    df['CCI'] = talib.CCI(
        df['high'], df['low'], df['close'], timeperiod=CCI_PERIOD
    )
    df['WilliamsR'] = talib.WILLR(
        df['high'], df['low'], df['close'], timeperiod=WILLR_PERIOD
    )
    slowk, slowd = talib.STOCH(df['high'], df['low'], df['close'],
                               fastk_period=STOCH_FASTK_PERIOD,
                               slowk_period=STOCH_SLOWK_PERIOD,
                               slowk_matype=0,
                               slowd_period=STOCH_SLOWD_PERIOD,
                               slowd_matype=0)
    df['Stochastic_SlowK'] = slowk
    df['Stochastic_SlowD'] = slowd
    df['MFI'] = talib.MFI(df['high'], df['low'], df['close'], df['volume'],
                          timeperiod=MFI_PERIOD)
    
    # Handle missing data
    df = df.ffill().bfill().fillna(0)
    df.dropna(inplace=True)

    # Convert back to timestamp after calculations
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
    
    return df