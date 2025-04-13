import pandas as pd
import numpy as np

def calculeaza_RSI(df, perioada=14):
    close = df["close"]
    delta = close.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=perioada).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=perioada).mean()
    rs = gain / loss
    df["rsi"] = 100 - (100 / (1 + rs))
    return df

def calculeaza_SMA(df, perioada=20):
    df["sma"] = df["close"].rolling(window=perioada).mean()
    return df

def calculeaza_EMA(df, perioada=20):
    df["ema"] = df["close"].ewm(span=perioada, adjust=False).mean()
    return df

def calculeaza_Bollinger(df, perioada=20):
    sma = df["close"].rolling(window=perioada).mean()
    std = df["close"].rolling(window=perioada).std()
    df["bollinger_upper"] = sma + (2 * std)
    df["bollinger_lower"] = sma - (2 * std)
    return df

def calculeaza_OBV(df):
    obv = [0]
    for i in range(1, len(df)):
        if df["close"][i] > df["close"][i - 1]:
            obv.append(obv[-1] + df["volume"][i])
        elif df["close"][i] < df["close"][i - 1]:
            obv.append(obv[-1] - df["volume"][i])
        else:
            obv.append(obv[-1])
    df["obv"] = obv
    return df

def calculeaza_StochRSI(df, perioada=14):
    rsi = df["rsi"]
    stoch_rsi = (rsi - rsi.rolling(window=perioada).min()) / (
        rsi.rolling(window=perioada).max() - rsi.rolling(window=perioada).min()
    )
    df["stochrsi"] = stoch_rsi
    return df

def calculeaza_MACD(df, short=12, long=26, signal=9):
    exp1 = df["close"].ewm(span=short, adjust=False).mean()
    exp2 = df["close"].ewm(span=long, adjust=False).mean()
    df["macd"] = exp1 - exp2
    df["signal"] = df["macd"].ewm(span=signal, adjust=False).mean()
    df["histogram"] = df["macd"] - df["signal"]
    return df

def adauga_toti_indicatorii(df):
    df = calculeaza_RSI(df)
    print("✔ rsi added", df.columns)
    df = calculeaza_SMA(df)
    print("✔ sma added", df.columns)
    df = calculeaza_EMA(df)
    print("✔ ema added", df.columns)
    df = calculeaza_Bollinger(df)
    print("✔ bollinger added", df.columns)
    df = calculeaza_OBV(df)
    print("✔ obv added", df.columns)
    df = calculeaza_MACD(df)
    print("✔ macd added", df.columns)
    df = calculeaza_StochRSI(df)
    print("✔ stochrsi added", df.columns)
    return df
