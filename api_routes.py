# api_routes.py
from flask import jsonify, request
from model_core import get_ai_prediction, SYMBOLS, MODELS, CONFIDENCE, LAST_UPDATE
from binance.client import Client
import os
import logging
import pandas as pd

# Load API keys
api_key = os.getenv("BINANCE_API_KEY")
api_secret = os.getenv("BINANCE_API_SECRET")
client = Client(api_key, api_secret)

def register_routes(app):

    @app.route("/api/price/<symbol>")
    def get_current_price(symbol):
        try:
            price = float(client.get_symbol_ticker(symbol=symbol.upper())['price'])
            return jsonify({"symbol": symbol.upper(), "price": round(price, 8 if price < 0.01 else 4)})
        except Exception as e:
            logging.error(f"Error fetching price for {symbol}: {e}")
            return jsonify({"error": "Price fetch failed"}), 500

    @app.route("/api/symbols")
    def get_symbols():
        return jsonify({"symbols": list(MODELS.keys())})

    @app.route("/api/predict/<symbol>")
    def predict(symbol):
        result = get_ai_prediction(symbol, client)
        if result is None:
            return jsonify({"error": "Prediction failed"}), 500
        return jsonify(result)

    @app.route("/")
    def health():
        return jsonify({
            "status": "healthy",
            "last_update": LAST_UPDATE.isoformat() if LAST_UPDATE else None,
            "symbols": len(SYMBOLS)
        })

    @app.route("/api/debug/available")
    def available_symbols_with_status():
        folder = "crypto_data"
        results = []

        for file in os.listdir(folder):
            if file.endswith(".csv"):
                symbol = file.replace(".csv", "")
                path = os.path.join(folder, file)
                try:
                    df = pd.read_csv(path)
                    if df.empty or len(df) < 30:
                        results.append({"symbol": symbol, "status": "too few rows"})
                        continue

                    required = {"open", "high", "low", "close", "volume"}
                    if not required.issubset(set(df.columns)):
                        results.append({"symbol": symbol, "status": "missing columns"})
                        continue

                    pred = get_ai_prediction(symbol, client)
                    if not pred or "current_price" not in pred or "predicted_price" not in pred:
                        results.append({"symbol": symbol, "status": "missing values"})
                    else:
                        results.append({"symbol": symbol, "status": "ok"})

                except Exception as e:
                    results.append({"symbol": symbol, "status": f"error: {str(e)}"})

        return jsonify(results)
