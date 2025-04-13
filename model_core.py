import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from dotenv import load_dotenv
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
import logging
from indicatori_tehnici import adauga_toti_indicatorii
from binance.client import Client

# Globals
SYMBOLS = []
MODELS = {}
SCALERS = {}
CONFIDENCE = {}
LAST_UPDATE = None
CSV_DIR = os.path.join(os.path.dirname(__file__), 'crypto_data')
PREDICTIONS_DIR = os.path.join(os.path.dirname(__file__), 'predictions')
os.makedirs(CSV_DIR, exist_ok=True)
os.makedirs(PREDICTIONS_DIR, exist_ok=True)

# Binance Client
load_dotenv()
api_key = os.getenv("BINANCE_API_KEY")
api_secret = os.getenv("BINANCE_API_SECRET")
client = Client(api_key, api_secret)

def prepare_data(df):
    try:
        df = df.copy()
        df.columns = [col.lower() for col in df.columns]
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df[['open', 'high', 'low', 'close', 'volume']] = df[['open', 'high', 'low', 'close', 'volume']].astype(float)
        df = adauga_toti_indicatorii(df)
        df['target'] = df['close'].rolling(window=24).mean().shift(-24)
        df = df.dropna()
        return df, df['target']
    except Exception as e:
        logging.error(f"prepare_data error: {e}")
        return None, None

def train_model(symbol):
    try:
        path = os.path.join(CSV_DIR, f"{symbol}.csv")
        if not os.path.exists(path): return
        df = pd.read_csv(path)
        X, y = prepare_data(df)
        if X is None or y is None: return
        y = y.dropna()
        X = X.loc[y.index]
        if len(y) < 50: return

        features = [
            'rsi', 'sma', 'ema', 'macd', 'signal', 'histogram',
            'bollinger_upper', 'bollinger_lower', 'obv', 'stochrsi',
            'close', 'volume'
        ]
        scaler = MinMaxScaler()
        X_scaled = scaler.fit_transform(X[features])
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_scaled, y)
        accuracy = compute_accuracy(symbol)
        MODELS[symbol] = model
        SCALERS[symbol] = scaler
        CONFIDENCE[symbol] = round(accuracy, 2)
        logging.info(f"Model trained for {symbol} | accuracy: {CONFIDENCE[symbol]}%")
    except Exception as e:
        logging.error(f"Error training model for {symbol}: {e}")

def compute_accuracy(symbol):
    path = os.path.join(PREDICTIONS_DIR, f"{symbol}.csv")
    if not os.path.exists(path): return 0.0
    try:
        df = pd.read_csv(path).dropna()
        df["error"] = np.abs(df["predicted_price"] - df["actual_price"])
        df["percent_error"] = (df["error"] / df["actual_price"]) * 100
        avg_error = df["percent_error"].tail(50).mean()
        return max(0.0, 100 - avg_error)
    except Exception as e:
        logging.warning(f"Accuracy calc failed for {symbol}: {e}")
        return 0.0

def save_prediction(symbol, timestamp, prediction, actual_price):
    path = os.path.join(PREDICTIONS_DIR, f"{symbol}.csv")
    row = {"timestamp": timestamp, "predicted_price": prediction, "actual_price": actual_price}
    try:
        if os.path.exists(path):
            df = pd.read_csv(path)
            df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
        else:
            df = pd.DataFrame([row])
        df = df.tail(100)
        df.to_csv(path, index=False)
        logging.info(f"Saved prediction for {symbol} at {timestamp}")
    except Exception as e:
        logging.warning(f"Failed to save prediction for {symbol}: {e}")

def get_prediction_24h_ago(symbol):
    path = os.path.join(PREDICTIONS_DIR, f"{symbol}.csv")
    if not os.path.exists(path):
        return None
    try:
        df = pd.read_csv(path)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        target_time = datetime.utcnow() - timedelta(hours=24)
        df["time_diff"] = (df["timestamp"] - target_time).abs()
        closest_row = df.loc[df["time_diff"].idxmin()]
        return float(closest_row["predicted_price"])
    except Exception as e:
        logging.warning(f"⚠️ Error getting 24h ago prediction for {symbol}: {e}")
        return None

def update_historical_data():
    global SYMBOLS, LAST_UPDATE
    try:
        SYMBOLS = [f.replace(".csv", "") for f in os.listdir(CSV_DIR) if f.endswith(".csv")]
        for symbol in SYMBOLS:
            try:
                klines = client.get_historical_klines(symbol, Client.KLINE_INTERVAL_1HOUR, "30 days ago UTC")
                df = pd.DataFrame(klines, columns=[
                    'timestamp', 'open', 'high', 'low', 'close', 'volume',
                    'close_time', 'quote_asset_volume', 'number_of_trades',
                    'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
                ])
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
                df = df.dropna().drop_duplicates(subset='timestamp').sort_values('timestamp')
                path = os.path.join(CSV_DIR, f"{symbol}.csv")
                if os.path.exists(path):
                    old_df = pd.read_csv(path)
                    old_df['timestamp'] = pd.to_datetime(old_df['timestamp'], errors='coerce')
                    old_df = old_df[old_df['timestamp'] < df['timestamp'].min()]
                    df = pd.concat([old_df, df])
                df.to_csv(path, index=False)
                train_model(symbol)
            except Exception as e:
                logging.warning(f"Error updating {symbol}: {e}")
        LAST_UPDATE = datetime.now()
    except Exception as e:
        logging.error(f"Global update error: {e}")

def get_ai_prediction(symbol, client):
    try:
        symbol = symbol.upper()
        if symbol not in MODELS or symbol not in SCALERS:
            return None
        path = os.path.join(CSV_DIR, f"{symbol}.csv")
        if not os.path.exists(path): return None
        df = pd.read_csv(path)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = adauga_toti_indicatorii(df).dropna()
        if df.empty: return None
        latest = df.iloc[-1]
        features = [
            latest['rsi'], latest['sma'], latest['ema'], latest['macd'], latest['signal'],
            latest['histogram'], latest['bollinger_upper'], latest['bollinger_lower'],
            latest['obv'], latest['stochrsi'], latest['close'], latest['volume']
        ]
        X = pd.DataFrame([features], columns=[
            'rsi', 'sma', 'ema', 'macd', 'signal', 'histogram',
            'bollinger_upper', 'bollinger_lower', 'obv', 'stochrsi',
            'close', 'volume'
        ])
        X = SCALERS[symbol].transform(X)
        preds = [tree.predict(X)[0] for tree in MODELS[symbol].estimators_]
        prediction = np.mean(preds)
        prediction_24h_ago = get_prediction_24h_ago(symbol)
        current_price = float(client.get_symbol_ticker(symbol=symbol)['price'])
        timestamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
        save_prediction(symbol, timestamp, prediction, current_price)
        trend = "up" if prediction > current_price else "down"
        confidence = CONFIDENCE.get(symbol, 0)
        profit = (prediction - current_price) / current_price * 10 if current_price else 0
        return {
            "symbol": symbol,
            "current_price": round(current_price, 8 if current_price < 0.01 else 4),
            "predicted_price": round(prediction, 8 if prediction < 0.01 else 4),
            "ai_confidence": round(confidence, 2),
            "trend": trend,
            "simulated_profit": round(profit, 2),
            "prediction_24h_ago": round(prediction_24h_ago, 8 if prediction_24h_ago and prediction_24h_ago < 0.01 else 4) if prediction_24h_ago else None,
        }
    except Exception as e:
        logging.error(f"Prediction error for {symbol}: {e}")
        return None
