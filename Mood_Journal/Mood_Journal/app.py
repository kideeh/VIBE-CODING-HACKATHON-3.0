# ----------------------------
# Imports at the top of app.py
# ----------------------------
import os
import requests
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
HF_API_KEY = os.getenv("HF_API_KEY")

# Hugging Face Sentiment Function
def analyze_sentiment(text):
    """
    Sends text to Hugging Face sentiment-analysis model.
    Returns label (POSITIVE/NEGATIVE/NEUTRAL) and score.
    """
    if not text or not HF_API_KEY:
        return "NEUTRAL", 0.0

    url = "https://api-inference.huggingface.co/models/distilbert-base-uncased-finetuned-sst-2-english"
    headers = {"Authorization": f"Bearer {HF_API_KEY}"}

    try:
        response = requests.post(url, headers=headers, json={"inputs": text})
        result = response.json()

        if isinstance(result, list) and len(result) > 0 and isinstance(result[0], list):
            label = result[0][0]["label"]
            score = result[0][0]["score"]
            return label, score
        elif isinstance(result, list) and "label" in result[0]:
            return result[0]["label"], result[0]["score"]
        else:
            return "NEUTRAL", 0.0
    except Exception as e:
        print("Sentiment API error:", e)
        return "NEUTRAL", 0.0
"""
app.py - Complete Mood Journal (single-file)
Safe to replace your previous app.py (back it up first).
"""

import os
import sqlite3
import json
import csv
import io
import re
from datetime import datetime, timedelta
from functools import wraps

from flask import (
    Flask, render_template, request, redirect, url_for, session,
    flash, send_file, jsonify, abort
)
from werkzeug.security import generate_password_hash, check_password_hash

# Optional imports — handled gracefully if missing
try:
    from cryptography.fernet import Fernet
except Exception:
    Fernet = None

try:
    from apscheduler.schedulers.background import BackgroundScheduler
except Exception:
    BackgroundScheduler = None

try:
    import jwt as pyjwt  # PyJWT (optional)
except Exception:
    pyjwt = None

try:
    import requests
except Exception:
    requests = None

# dotenv (optional)
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

# -------------------------
# Config
# -------------------------
APP_SECRET = os.getenv("APP_SECRET", "dev_app_secret_change_me")
JWT_SECRET = os.getenv("JWT_SECRET", "dev_jwt_secret_change_me")
HF_API_TOKEN = os.getenv("HF_API_TOKEN", None)
HF_MODEL = os.getenv("HF_MODEL", "distilbert-base-uncased-finetuned-sst-2-english")
FERNET_KEY_ENV = os.getenv("FERNET_KEY", None)  # optional base64 key

# Flask app
app = Flask(__name__, template_folder="templates", static_folder="static")
app.secret_key = APP_SECRET

# -------------------------
# Paths & DB setup
# -------------------------
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
os.makedirs(DATA_DIR, exist_ok=True)  # ensure data folder exists
DB_PATH = os.path.join(DATA_DIR, "mood_journal.db")

def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_db()
    c = conn.cursor()
    # Users
    c.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT,
            email TEXT UNIQUE,
            password_hash TEXT,
            avatar TEXT,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)
    # Entries
    c.execute("""
        CREATE TABLE IF NOT EXISTS entries (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            mood_rating INTEGER,
            emotions TEXT,
            energy_level INTEGER,
            stress_level INTEGER,
            notes_encrypted TEXT,
            coping TEXT,
            sentiment TEXT,
            FOREIGN KEY(user_id) REFERENCES users(id) ON DELETE CASCADE
        )
    """)
    conn.commit()
    conn.close()

# initialize DB on startup
init_db()

# -------------------------
# Fernet encryption (persistent key)
# -------------------------
FERNET_KEY_PATH = os.path.join(DATA_DIR, "fernet.key")

def get_or_create_fernet():
    if Fernet is None:
        print("[WARN] cryptography not installed — encryption disabled.")
        return None
    # priority: explicit env key (FERNET_KEY), else persistent file, else generate & save
    if FERNET_KEY_ENV:
        key = FERNET_KEY_ENV.encode() if isinstance(FERNET_KEY_ENV, str) else FERNET_KEY_ENV
        try:
            return Fernet(key)
        except Exception:
            pass
    # try persistent file
    if os.path.exists(FERNET_KEY_PATH):
        try:
            with open(FERNET_KEY_PATH, "rb") as f:
                key = f.read().strip()
                return Fernet(key)
        except Exception:
            pass
    # generate & save key
    try:
        key = Fernet.generate_key()
        with open(FERNET_KEY_PATH, "wb") as f:
            f.write(key)
        return Fernet(key)
    except Exception:
        print("[WARN] Failed to create Fernet key.")
        return None

fernet = get_or_create_fernet()

def encrypt_text(plain: str):
    if not plain:
        return None
    if not fernet:
        return None
    try:
        return fernet.encrypt(plain.encode()).decode()
    except Exception as e:
        print("[ERROR] encrypt_text:", e)
        return None

def decrypt_text(token: str):
    if not token:
        return None
    if not fernet:
        return None
    try:
        return fernet.decrypt(token.encode()).decode()
    except Exception:
        return "[decryption error]"

# -------------------------
# Simple helpers / auth
# -------------------------
def login_required(fn):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        if "user_id" not in session:
            flash("Please log in.", "warning")
            return redirect(url_for("login"))
        return fn(*args, **kwargs)
    return wrapper

# -------------------------
# Hugging Face sentiment helper
# -------------------------
def analyze_sentiment_hf(text: str):
    """Return 'LABEL (0.95)' or fallback."""
    if not text:
        return ""
    s = text.strip()
    if not s:
        return ""
    # If HF token missing or requests unavailable -> basic heuristic fallback
    if not HF_API_TOKEN or requests is None:
        low = s.lower()
        if any(w in low for w in ["happy", "joy", "amazing", "excited", "great", "good"]):
            return "POSITIVE (0.95)"
        if any(w in low for w in ["sad", "depressed", "angry", "upset", "bad", "down"]):
            return "NEGATIVE (0.90)"
        return "NEUTRAL"
    url = f"https://api-inference.huggingface.co/models/{HF_MODEL}"
    headers = {"Authorization": f"Bearer {HF_API_TOKEN}"}
    payload = {"inputs": s}
    try:
        resp = requests.post(url, headers=headers, json=payload, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        # typical HF output: [{'label':'POSITIVE','score':0.99}]
        if isinstance(data, list) and data and isinstance(data[0], dict):
            label = data[0].get("label", "")
            score = data[0].get("score", 0.0)
            return f"{label} ({float(score):.2f})"
        return str(data)[:200]
    except Exception as e:
        print("[WARN] HF inference failed:", e)
        return "Unknown"

# -------------------------
# Keyword extraction for insights
# -------------------------
STOPWORDS = set([
    "the","and","for","that","with","you","this","was","but","are","have","not","they","from",
    "what","when","which","their","there","been","has","one","all","any","can","were","had","she",
    "him","her","his","its","our","out","who","how","a","i","to","in","of","on","is","it"
])

def extract_keywords(text: str, top_n=10):
    if not text:
        return []
    s = re.sub(r'[^a-z0-9\s]', ' ', text.lower())
    words = [w for w in s.split() if len(w) > 2 and w not in STOPWORDS]
    freq = {}
    for w in words:
        freq[w] = freq.get(w, 0) + 1
    items = sorted(freq.items(), key=lambda kv: -kv[1])
    return [k for k, _ in items[:top_n]]

# -------------------------
# Scheduler (optional)
# -------------------------
scheduler = None
if BackgroundScheduler:
    try:
        scheduler = BackgroundScheduler()
        def _reminder_job():
            # placeholder: implement per-user reminders
            print("[scheduler] reminder job tick")
        scheduler.add_job(_reminder_job, "interval", hours=24, id="daily_reminder", replace_existing=True)
        scheduler.start()
    except Exception as e:
        print("[WARN] scheduler failed to start:", e)
else:
    print("[INFO] APScheduler not present; reminders disabled.")

# -------------------------
# Routes: Auth & Profile
# -------------------------
@app.route("/signup", methods=["GET", "POST"])
def signup():
    if request.method == "POST":
        name = (request.form.get("name") or "").strip()
        email = (request.form.get("email") or "").strip().lower()
        password = request.form.get("password") or ""
        if not email or not password:
            flash("Email and password required.", "danger")
            return redirect(url_for("signup"))
        pw_hash = generate_password_hash(password)
        conn = get_db()
        c = conn.cursor()
        try:
            c.execute("INSERT INTO users (name, email, password_hash) VALUES (?, ?, ?)", (name, email, pw_hash))
            conn.commit()
            conn.close()
            flash("Signup successful — please log in.", "success")
            return redirect(url_for("login"))
        except sqlite3.IntegrityError:
            conn.close()
            flash("Email already registered.", "danger")
            return redirect(url_for("signup"))
    return render_template("signup.html")

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        email = (request.form.get("email") or "").strip().lower()
        password = request.form.get("password") or ""
        conn = get_db()
        c = conn.cursor()
        c.execute("SELECT id, name, password_hash FROM users WHERE email=?", (email,))
        row = c.fetchone()
        conn.close()
        if not row or not check_password_hash(row["password_hash"], password):
            flash("Invalid credentials.", "danger")
            return redirect(url_for("login"))
        session.clear()
        session["user_id"] = row["id"]
        session["user_name"] = row["name"]
        flash("Welcome back!", "success")
        return redirect(url_for("dashboard"))
    return render_template("login.html")

@app.route("/logout")
def logout():
    session.clear()
    flash("Logged out.", "info")
    return redirect(url_for("login"))

@app.route("/profile", methods=["GET", "POST"])
@login_required
def profile():
    user_id = session["user_id"]
    conn = get_db()
    c = conn.cursor()
    if request.method == "POST":
        name = (request.form.get("name") or "").strip()
        email = (request.form.get("email") or "").strip().lower()
        password = request.form.get("password") or ""
        avatar = (request.form.get("avatar") or "").strip() or None
        if password:
            pw_hash = generate_password_hash(password)
            c.execute("UPDATE users SET name=?, email=?, password_hash=?, avatar=? WHERE id=?", (name, email, pw_hash, avatar, user_id))
        else:
            c.execute("UPDATE users SET name=?, email=?, avatar=? WHERE id=?", (name, email, avatar, user_id))
        conn.commit()
        conn.close()
        session["user_name"] = name
        flash("Profile updated.", "success")
        return redirect(url_for("profile"))
    c.execute("SELECT id, name, email, avatar, created_at FROM users WHERE id=?", (user_id,))
    user = c.fetchone()
    conn.close()
    return render_template("profile.html", user=user)

@app.route("/delete-account", methods=["GET", "POST"])
@login_required
def delete_account():
    user_id = session["user_id"]
    if request.method == "POST":
        conn = get_db()
        c = conn.cursor()
        c.execute("DELETE FROM entries WHERE user_id=?", (user_id,))
        c.execute("DELETE FROM users WHERE id=?", (user_id,))
        conn.commit()
        conn.close()
        session.clear()
        flash("Account deleted and data removed.", "info")
        return redirect(url_for("signup"))
    return render_template("delete.html")

# -------------------------
# Entries CRUD
# -------------------------
@app.route("/", methods=["GET", "POST"])
@login_required
def index():
    # POST => create entry
    if request.method == "POST":
        user_id = session["user_id"]
        # mood_rating (1-10)
        try:
            mood_rating = int(request.form.get("mood_rating", 5))
        except Exception:
            mood_rating = 5
        # emotions list
        emotions = request.form.getlist("emotions") or []
        custom_emotion = (request.form.get("emotions_custom") or "").strip()
        if custom_emotion:
            emotions.append(custom_emotion)
        emotions_json = json.dumps(emotions) if emotions else None
        # energy & stress
        try:
            energy_level = int(request.form.get("energy_level", 5))
        except Exception:
            energy_level = 5
        try:
            stress_level = int(request.form.get("stress_level", 5))
        except Exception:
            stress_level = 5
        notes = (request.form.get("notes") or "").strip()
        notes_encrypted = encrypt_text(notes) if notes else None
        coping_raw = (request.form.get("coping") or "").strip()
        coping_list = [s.strip() for s in coping_raw.split(",") if s.strip()] if coping_raw else []
        coping_json = json.dumps(coping_list) if coping_list else None

        # sentiment
        sentiment_label = analyze_sentiment_hf(notes)

        conn = get_db()
        c = conn.cursor()
        c.execute("""
            INSERT INTO entries (user_id, mood_rating, emotions, energy_level, stress_level, notes_encrypted, coping, sentiment)
            VALUES (?,?,?,?,?,?,?,?)
        """, (user_id, mood_rating, emotions_json, energy_level, stress_level, notes_encrypted, coping_json, sentiment_label))
        conn.commit()
        conn.close()
        flash("✅ Mood entry saved!", "success")
        return redirect(url_for("dashboard"))
    # GET => new-entry form
    return render_template("index.html")
@app.route("/add_entry", methods=["POST"])
def add_entry():
    user_id = session.get("user_id")
    if not user_id:
        return redirect(url_for("login"))

    mood = request.form.get("mood")
    emotions = request.form.getlist("emotions")
    energy = request.form.get("energy")
    stress = request.form.get("stress")
    notes = request.form.get("notes")

    # <<< NEW: analyze sentiment >>>
    sentiment_label, sentiment_score = analyze_sentiment(notes)

    # Save to DB
    conn = get_db()
    c = conn.cursor()
    c.execute("""
        INSERT INTO entries (user_id, mood, emotions, energy, stress, notes, sentiment, sentiment_score, timestamp)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, datetime('now'))
    """, (user_id, mood, ",".join(emotions), energy, stress, notes, sentiment_label, sentiment_score))
    conn.commit()
    conn.close()

    flash("✅ Mood entry saved!", "success")
    return redirect(url_for("dashboard"))

@app.route("/entries")
@login_required
def entries_page():
    user_id = session["user_id"]
    conn = get_db()
    c = conn.cursor()
    c.execute("SELECT id, created_at, mood_rating, emotions, energy_level, stress_level, notes_encrypted, coping, sentiment FROM entries WHERE user_id=? ORDER BY created_at DESC", (user_id,))
    rows = c.fetchall()
    conn.close()
    entries = []
    for r in rows:
        notes_plain = decrypt_text(r["notes_encrypted"]) if r["notes_encrypted"] else ""
        emotions = json.loads(r["emotions"]) if r["emotions"] else []
        coping = json.loads(r["coping"]) if r["coping"] else []
        entries.append({
            "id": r["id"],
            "created_at": r["created_at"],
            "mood_rating": r["mood_rating"],
            "emotions": emotions,
            "energy_level": r["energy_level"],
            "stress_level": r["stress_level"],
            "notes": notes_plain,
            "coping": coping,
            "sentiment": r["sentiment"]
        })
    return render_template("entries.html", entries=entries)

@app.route("/entry/<int:eid>/edit", methods=["GET", "POST"])
@login_required
def edit_entry(eid):
    user_id = session["user_id"]
    conn = get_db()
    c = conn.cursor()
    c.execute("SELECT * FROM entries WHERE id=? AND user_id=?", (eid, user_id))
    row = c.fetchone()
    if not row:
        conn.close()
        flash("Entry not found.", "warning")
        return redirect(url_for("entries_page"))
    if request.method == "POST":
        try:
            mood_rating = int(request.form.get("mood_rating", row["mood_rating"] or 5))
        except Exception:
            mood_rating = row["mood_rating"] or 5
        emotions = request.form.getlist("emotions") or []
        custom_emotion = (request.form.get("emotions_custom") or "").strip()
        if custom_emotion:
            emotions.append(custom_emotion)
        emotions_json = json.dumps(emotions) if emotions else None
        try:
            energy_level = int(request.form.get("energy_level", row["energy_level"] or 5))
        except Exception:
            energy_level = row["energy_level"] or 5
        try:
            stress_level = int(request.form.get("stress_level", row["stress_level"] or 5))
        except Exception:
            stress_level = row["stress_level"] or 5
        notes = (request.form.get("notes") or "")
        notes_encrypted = encrypt_text(notes) if notes else None
        coping_raw = (request.form.get("coping") or "")
        coping_list = [s.strip() for s in coping_raw.split(",") if s.strip()] if coping_raw else []
        coping_json = json.dumps(coping_list) if coping_list else None
        sentiment_label = analyze_sentiment_hf(notes)
        c.execute("""
            UPDATE entries SET mood_rating=?, emotions=?, energy_level=?, stress_level=?, notes_encrypted=?, coping=?, sentiment=?
            WHERE id=? AND user_id=?
        """, (mood_rating, emotions_json, energy_level, stress_level, notes_encrypted, coping_json, sentiment_label, eid, user_id))
        conn.commit()
        conn.close()
        flash("Entry updated.", "success")
        return redirect(url_for("entries_page"))
    # GET - prepare form
    notes_plain = decrypt_text(row["notes_encrypted"]) if row["notes_encrypted"] else ""
    entry = {
        "id": row["id"],
        "mood_rating": row["mood_rating"],
        "emotions": json.loads(row["emotions"]) if row["emotions"] else [],
        "energy_level": row["energy_level"],
        "stress_level": row["stress_level"],
        "notes": notes_plain,
        "coping": json.loads(row["coping"]) if row["coping"] else [],
        "sentiment": row["sentiment"],
        "created_at": row["created_at"]
    }
    conn.close()
    return render_template("edit_entry.html", entry=entry)

@app.route("/entry/<int:eid>/delete", methods=["POST"])
@login_required
def delete_entry(eid):
    user_id = session["user_id"]
    conn = get_db()
    c = conn.cursor()
    c.execute("DELETE FROM entries WHERE id=? AND user_id=?", (eid, user_id))
    conn.commit()
    conn.close()
    flash("Entry deleted.", "info")
    return redirect(url_for("entries_page"))

# -------------------------
# Dashboard & analytics
# -------------------------
@app.route("/dashboard")
@login_required
def dashboard():
    user_id = session["user_id"]
    conn = get_db()
    c = conn.cursor()
    c.execute("SELECT created_at, mood_rating, emotions FROM entries WHERE user_id=? ORDER BY created_at ASC", (user_id,))
    rows = c.fetchall()
    conn.close()
    labels = [r["created_at"] for r in rows]
    data = [r["mood_rating"] for r in rows]
    emotion_counts = {}
    for r in rows:
        try:
            emos = json.loads(r["emotions"]) if r["emotions"] else []
        except Exception:
            emos = []
        for e in emos:
            emotion_counts[e] = emotion_counts.get(e, 0) + 1
    avg_mood = round(sum(data)/len(data), 2) if data else None
    best_day = None
    worst_day = None
    if rows:
        best = max(rows, key=lambda x: x["mood_rating"])
        worst = min(rows, key=lambda x: x["mood_rating"])
        best_day = best["created_at"]
        worst_day = worst["created_at"]
    # keywords from notes
    conn = get_db()
    c = conn.cursor()
    c.execute("SELECT notes_encrypted FROM entries WHERE user_id=?", (user_id,))
    notes_rows = c.fetchall()
    conn.close()
    notes_texts = []
    for n in notes_rows:
        if n["notes_encrypted"]:
            try:
                notes_texts.append(decrypt_text(n["notes_encrypted"]))
            except Exception:
                pass
    all_notes = "\n".join([t for t in notes_texts if t])
    keywords = extract_keywords(all_notes, top_n=10)
    # render
    return render_template("dashboard.html", labels=labels, data=data, emotion_counts=emotion_counts,
                           avg_mood=avg_mood, best_day=best_day, worst_day=worst_day, keywords=keywords)

# Chart data API for Chart.js
@app.route("/api/entries_json")
@login_required
def entries_json():
    user_id = session["user_id"]
    conn = get_db()
    c = conn.cursor()
    c.execute("SELECT created_at, mood_rating FROM entries WHERE user_id=? ORDER BY created_at ASC", (user_id,))
    rows = c.fetchall()
    conn.close()
    labels = [r["created_at"] for r in rows]
    values = [r["mood_rating"] for r in rows]
    return jsonify({"labels": labels, "values": values})

# -------------------------
# Export CSV
# -------------------------
@app.route("/export")
@login_required
def export_csv():
    user_id = session["user_id"]
    conn = get_db()
    c = conn.cursor()
    c.execute("SELECT created_at,mood_rating,emotions,energy_level,stress_level,notes_encrypted,coping,sentiment FROM entries WHERE user_id=? ORDER BY created_at DESC", (user_id,))
    rows = c.fetchall()
    conn.close()
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["Timestamp","MoodRating","Emotions","Energy","Stress","Notes","Coping","Sentiment"])
    for r in rows:
        notes_plain = decrypt_text(r["notes_encrypted"]) if r["notes_encrypted"] else ""
        writer.writerow([r["created_at"], r["mood_rating"], r["emotions"], r["energy_level"], r["stress_level"], notes_plain, r["coping"], r["sentiment"]])
    mem = io.BytesIO()
    mem.write(output.getvalue().encode("utf-8"))
    mem.seek(0)
    filename = f"mood_export_{datetime.utcnow().strftime('%Y%m%d%H%M%S')}.csv"
    return send_file(mem, as_attachment=True, download_name=filename, mimetype="text/csv")

# HF test endpoint (optional)
@app.route("/hf_emotion", methods=["POST"])
@login_required
def hf_emotion():
    text = (request.json or {}).get("text", "")
    if not text:
        return jsonify({"error": "missing text"}), 400
    return jsonify({"result": analyze_sentiment_hf(text)})

# API token (JWT) endpoint (optional)
@app.route("/api/token", methods=["POST"])
def api_token():
    if pyjwt is None:
        return jsonify({"error": "PyJWT not installed"}), 500
    data = request.json or {}
    email = data.get("email")
    password = data.get("password")
    if not email or not password:
        return jsonify({"error": "missing credentials"}), 400
    conn = get_db()
    c = conn.cursor()
    c.execute("SELECT id, password_hash FROM users WHERE email=?", (email.lower(),))
    row = c.fetchone()
    conn.close()
    if not row or not check_password_hash(row["password_hash"], password):
        return jsonify({"error": "invalid credentials"}), 401
    payload = {"sub": row["id"], "exp": int((datetime.utcnow() + timedelta(hours=2)).timestamp())}
    token = pyjwt.encode(payload, JWT_SECRET, algorithm="HS256")
    return jsonify({"access_token": token})

# -------------------------
# Health
# -------------------------
@app.route("/health")
def health():
    return jsonify({"ok": True, "db": os.path.exists(DB_PATH)})

# -------------------------
# Run
# -------------------------
if __name__ == "__main__":
    print("[INFO] Mood Journal starting")
    print(f"[INFO] DB_PATH = {DB_PATH}")
    if Fernet is None:
        print("[WARN] cryptography not installed — notes encryption disabled.")
    if requests is None:
        print("[WARN] requests not installed — HF integration disabled (fallback will be used).")
    if pyjwt is None:
        print("[WARN] PyJWT not installed — /api/token disabled.")
    app.run(debug=True)
