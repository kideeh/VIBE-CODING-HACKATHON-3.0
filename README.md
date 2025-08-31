Mood Journal App

Track, Understand, and Improve Your Mood

Table of Contents

Overview

Features

Tech Stack

Installation

Usage

Screenshots

Contributing

License

Overview

The Mood Journal App is a full-stack web application that helps users log their moods, track emotions, analyze trends, and gain insights into their mental well-being.

Features

Authentication: Signup, Login, Logout, Profile management, Delete account

Daily Mood Entries: Mood rating, Emotion selection, Energy & stress level, Notes, Coping strategies

Dashboard: Mood trends, emotion frequency charts, weekly/monthly summaries

Export: CSV / PDF of your mood data

Advanced Features: Notifications/reminders, Dark/Light mode, AI-powered insights, Custom emotion tags

Tech Stack

Frontend: React + Tailwind CSS

Backend: Flask / FastAPI (Python-based)

Database: PostgreSQL (or SQLite for local dev)

Charts: Chart.js

Authentication: JWT / Session-based

Installation

Clone the repo:

git clone https://github.com/<your-username>/mood-journal.git
cd mood-journal


Create a virtual environment:

python -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate


Install dependencies:

pip install -r requirements.txt


Create .env file for sensitive config (JWT secret, API keys, etc).

Run the app:

python app.py


Open http://127.0.0.1:5000
 in your browser.

Usage

Sign up or log in

Add daily mood entries

View trends and insights in the dashboard

Export your mood data

Contributing

Fork the repository

Create a feature branch (git checkout -b feature-name)

Commit your changes (git commit -m "Add feature")

Push to the branch (git push origin feature-name)

Create a Pull Request
