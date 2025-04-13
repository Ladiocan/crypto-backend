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
