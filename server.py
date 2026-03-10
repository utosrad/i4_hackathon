"""
OpenSky proxy server: loads credentials from .env, gets OAuth2 token,
proxies /flights/arrival for LIVE FLIGHTS. Serves static files so the app
runs at http://localhost:8080/

Run: pip install -r requirements.txt && python server.py
Copy .env.example to .env and set OPENSKY_CLIENT_ID, OPENSKY_CLIENT_SECRET.
"""

import os
import time
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv
from flask import Flask, jsonify, send_from_directory
import requests

load_dotenv()

CLIENT_ID = os.environ.get("OPENSKY_CLIENT_ID")
CLIENT_SECRET = os.environ.get("OPENSKY_CLIENT_SECRET")
if not CLIENT_ID or not CLIENT_SECRET:
    print(
        "WARNING: OPENSKY_CLIENT_ID or OPENSKY_CLIENT_SECRET not set. "
        "/api/live-flights will return 503. Copy .env.example to .env and set credentials."
    )

TOKEN_URL = "https://auth.opensky-network.org/auth/realms/opensky-network/protocol/openid-connect/token"
ARRIVAL_URL = "https://opensky-network.org/api/flights/arrival"

app = Flask(__name__, static_folder=None)

_token = None
_token_expires = 0
TOKEN_BUFFER_SEC = 60


def get_token():
    global _token, _token_expires
    if _token and time.time() < _token_expires:
        return _token
    r = requests.post(
        TOKEN_URL,
        data={
            "grant_type": "client_credentials",
            "client_id": CLIENT_ID,
            "client_secret": CLIENT_SECRET,
        },
        headers={"Content-Type": "application/x-www-form-urlencoded"},
        timeout=10,
    )
    r.raise_for_status()
    data = r.json()
    _token = data["access_token"]
    _token_expires = time.time() + data.get("expires_in", 1800) - TOKEN_BUFFER_SEC
    return _token


@app.after_request
def cors(response):
    response.headers["Access-Control-Allow-Origin"] = "*"
    return response


@app.route("/api/live-flights")
def live_flights():
    if not CLIENT_ID or not CLIENT_SECRET:
        return jsonify({"error": "no_credentials", "message": "OpenSky credentials not configured."}), 503
    try:
        token = get_token()
    except Exception as e:
        return jsonify({"error": "token_failed", "message": str(e)}), 502
    now = datetime.now(timezone.utc)
    end_ts = int(now.timestamp())
    begin_ts = max(0, end_ts - 7200)
    url = f"{ARRIVAL_URL}?airport=CYYZ&begin={begin_ts}&end={end_ts}"
    try:
        r = requests.get(
            url,
            headers={"Authorization": f"Bearer {token}"},
            timeout=15,
        )
        r.raise_for_status()
        data = r.json()
        if not isinstance(data, list):
            return jsonify({"error": "invalid_response"}), 502
        return jsonify(data)
    except requests.RequestException as e:
        return jsonify({"error": "api_failed", "message": str(e)}), 503


@app.route("/")
def index():
    return send_from_directory(Path(__file__).resolve().parent, "index.html")


@app.route("/<path:path>")
def static_file(path):
    return send_from_directory(Path(__file__).resolve().parent, path)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port, debug=False)
