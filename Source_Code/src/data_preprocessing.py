import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def preprocess_data(input_path, output_path):
    df = pd.read_csv(input_path)
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date').set_index('Date')
    df.fillna(method='ffill', inplace=True)

    scaler = MinMaxScaler()
    df[df.columns] = scaler.fit_transform(df[df.columns])

    df.to_csv(output_path)
    return df
