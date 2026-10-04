"""AI Integration Platform - simple base version.

One small API that puts several AI providers behind a single endpoint.
Add a new provider by writing one function and registering it in PROVIDERS.
"""
import os

import requests
from dotenv import load_dotenv
from flask import Flask, jsonify, request, send_from_directory

load_dotenv()
app = Flask(__name__, static_folder="static")
TIMEOUT = 60


# ---------- Providers: each takes a message, returns reply text ----------
def echo_provider(message: str) -> str:
    """Offline test provider; works with no API key."""
    return f"Echo: {message}"


def anthropic_provider(message: str) -> str:
    r = requests.post(
        "https://api.anthropic.com/v1/messages",
        headers={
            "x-api-key": os.environ["ANTHROPIC_API_KEY"],
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        },
        json={
            "model": os.getenv("ANTHROPIC_MODEL", "claude-sonnet-5-5"),
            "max_tokens": 1000,
            "messages": [{"role": "user", "content": message}],
        },
        timeout=TIMEOUT,
    )
    r.raise_for_status()
    return "".join(b.get("text", "") for b in r.json()["content"])


def openai_provider(message: str) -> str:
    r = requests.post(
        "https://api.openai.com/v1/chat/completions",
        headers={"Authorization": f"Bearer {os.environ['OPENAI_API_KEY']}"},
        json={
            "model": os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            "messages": [{"role": "user", "content": message}],
        },
        timeout=TIMEOUT,
    )
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"]


# name -> (label, handler, env var holding the key or None)
PROVIDERS = {
    "echo": ("Echo (test)", echo_provider, None),
    "anthropic": ("Claude", anthropic_provider, "ANTHROPIC_API_KEY"),
    "openai": ("OpenAI", openai_provider, "OPENAI_API_KEY"),
}


def is_ready(env_key):
    return env_key is None or bool(os.getenv(env_key))


# ---------- Routes ----------
@app.get("/")
def index():
    return send_from_directory("static", "index.html")


@app.get("/api/health")
def health():
    return jsonify(status="ok")


@app.get("/api/providers")
def providers():
    return jsonify([
        {"id": pid, "label": label, "ready": is_ready(env)}
        for pid, (label, _, env) in PROVIDERS.items()
    ])


@app.post("/api/chat")
def chat():
    data = request.get_json(silent=True) or {}
    pid, message = data.get("provider", "echo"), (data.get("message") or "").strip()

    if pid not in PROVIDERS:
        return jsonify(error=f"Unknown provider '{pid}'."), 400
    if not message:
        return jsonify(error="Message is empty. Type something to send."), 400

    label, handler, env = PROVIDERS[pid]
    if not is_ready(env):
        return jsonify(error=f"{label} needs {env} in your .env file."), 400

    try:
        return jsonify(provider=pid, reply=handler(message))
    except requests.RequestException as exc:
        return jsonify(error=f"{label} request failed: {exc}"), 502


if __name__ == "__main__":
    app.run(debug=True, port=5000)
