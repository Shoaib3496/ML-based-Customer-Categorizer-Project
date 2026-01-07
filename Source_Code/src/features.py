def add_features(df):
    df['SMA_7'] = df['Close'].rolling(7).mean()
    df['SMA_14'] = df['Close'].rolling(14).mean()
    df['Volatility'] = df['Close'].rolling(14).std()
    df['Volume_MarketCap'] = df['Volume'] / df['MarketCap']
    return df.dropna()
