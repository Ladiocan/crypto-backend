from flask import Flask
from flask_cors import CORS
import logging
from threading import Thread
import time
import schedule
from api_routes import register_routes
from model_core import update_historical_data

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Flask app
app = Flask(__name__)
CORS(app)
register_routes(app)

def run_scheduler():
    while True:
        try:
            schedule.run_pending()
            time.sleep(1)
        except Exception as e:
            logging.error(f"Scheduler error: {e}")
            time.sleep(5)

def start():
    update_historical_data()
    schedule.every(1).hours.do(update_historical_data)
    run_scheduler()

if __name__ == "__main__":
    Thread(target=start, daemon=True).start()
    import os
port = int(os.environ.get("PORT", 5000))
app.run(host="0.0.0.0", port=port)

