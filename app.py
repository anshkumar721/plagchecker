import nltk
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import string
from nltk.corpus import stopwords
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from flask import Flask, request, jsonify, render_template
import requests
import json
import sqlite3
from datetime import datetime
import hashlib

# Download NLTK resources
try:
    nltk.data.find('corpora/stopwords')
    nltk.data.find('tokenizers/punkt')
except nltk.downloader.DownloadError:
    print("Downloading NLTK popular resources...")
    nltk.download("popular")
    print("NLTK popular resources downloaded.")

app = Flask(__name__)


# Initialize SQLite database
def init_db():
    conn = sqlite3.connect('plagiarism.db')
    c = conn.cursor()

    # Create history table
    c.execute('''CREATE TABLE IF NOT EXISTS checks
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  user_id TEXT,
                  text TEXT,
                  is_plagiarized INTEGER,
                  similarity_score REAL,
                  message TEXT,
                  sources TEXT,
                  timestamp DATETIME)''')

    # Create users table (for future authentication)
    c.execute('''CREATE TABLE IF NOT EXISTS users
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  username TEXT UNIQUE,
                  password_hash TEXT)''')

    conn.commit()
    conn.close()


init_db()


# Database helper functions
def save_to_db(user_id, text, is_plagiarized, similarity_score, message, sources):
    conn = sqlite3.connect('plagiarism.db')
    c = conn.cursor()

    c.execute('''INSERT INTO checks 
                 (user_id, text, is_plagiarized, similarity_score, message, sources, timestamp)
                 VALUES (?, ?, ?, ?, ?, ?, ?)''',
              (user_id, text, int(is_plagiarized), similarity_score, message, json.dumps(sources), datetime.now()))

    conn.commit()
    conn.close()


def get_history(user_id, limit=10):
    conn = sqlite3.connect('plagiarism.db')
    c = conn.cursor()

    c.execute('''SELECT text, is_plagiarized, similarity_score, message, sources, timestamp 
                 FROM checks 
                 WHERE user_id = ? 
                 ORDER BY timestamp DESC 
                 LIMIT ?''', (user_id, limit))

    results = c.fetchall()
    conn.close()

    history = []
    for row in results:
        history.append({
            'text': row[0],
            'is_plagiarized': bool(row[1]),
            'similarity_score': row[2],
            'message': row[3],
            'sources': json.loads(row[4]) if row[4] else [],
            'timestamp': row[5]
        })

    return history


# Google Custom Search API configuration
GOOGLE_SEARCH_API_KEY = "AIzaSyAojuCB9PI0SK2CldlNUjNmaf_IGDz7bd8"
CUSTOM_SEARCH_ENGINE_ID = "c463b455c47984210"
GOOGLE_SEARCH_URL = "https://www.googleapis.com/customsearch/v1"


def search_web_for_plagiarism(query_text):
    if not GOOGLE_SEARCH_API_KEY or not CUSTOM_SEARCH_ENGINE_ID:
        print("Warning: Google Search API key or CSE ID not configured. Web search skipped.")
        return []

    params = {
        "key": GOOGLE_SEARCH_API_KEY,
        "cx": CUSTOM_SEARCH_ENGINE_ID,
        "q": query_text,
        "num": 3
    }

    try:
        response = requests.get(GOOGLE_SEARCH_URL, params=params, timeout=5)
        response.raise_for_status()
        search_results = response.json()

        sources = []
        if 'items' in search_results:
            for item in search_results['items']:
                link = item.get('link')
                if link:
                    sources.append(link)
        return sources
    except Exception as e:
        print(f"Error during web search: {e}")
        return []


# Text preprocessing
def preprocess_text(text):
    text = str(text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = text.lower()
    stop_words = set(stopwords.words("english"))
    text = " ".join(word for word in text.split() if word not in stop_words)
    return text


# Model and vectorizer
loaded_model = None
loaded_vectorizer = None
X_all = None


def load_model_and_data():
    global loaded_model, loaded_vectorizer, X_all
    try:
        loaded_model, loaded_vectorizer = joblib.load('plagiarism_model.pkl')
        print("Model and vectorizer loaded from plagiarism_model.pkl")

        data = pd.read_csv(
            "article50.csv",
            sep=',',
            quotechar='"',
            header=0,
            on_bad_lines='warn'
        )

        data.columns = data.columns.str.strip().str.lower()
        data["source_text"] = data["source_text"].fillna('').astype(str).apply(preprocess_text)
        data["plagiarized_text"] = data["plagiarized_text"].fillna('').astype(str).apply(preprocess_text)

        X_all = loaded_vectorizer.transform(data["source_text"] + " " + data["plagiarized_text"])
        print("Dataset loaded and vectorized for cosine similarity calculation.")

    except Exception as e:
        print(f"Error loading model/data: {e}")
        loaded_model = None
        loaded_vectorizer = None
        X_all = None


with app.app_context():
    load_model_and_data()


# Routes
@app.route('/')
def index():
    return render_template('index.html')


@app.route('/check-plagiarism', methods=['POST'])
def check_plagiarism():
    try:
        text_to_check = request.json.get('text', '')
        user_id = request.json.get('user_id', 'anonymous')  # Default to 'anonymous' if no user_id provided

        if not text_to_check:
            return jsonify(
                {'is_plagiarized': False, 'similarity_score': 0, 'message': 'No text provided.', 'sources': []})

        if loaded_model is None or loaded_vectorizer is None or X_all is None:
            return jsonify({'is_plagiarized': False, 'similarity_score': 0,
                            'message': 'Plagiarism check service not available.', 'sources': []})

        preprocessed_text = preprocess_text(text_to_check)
        text_vector = loaded_vectorizer.transform([preprocessed_text])
        prediction_label = loaded_model.predict(text_vector)[0]
        cosine_similarity_score = cosine_similarity(text_vector, X_all).max() * 100

        is_plagiarized_result = bool(prediction_label == 1)
        message = "The text is likely original."

        if cosine_similarity_score > 75:
            message = f"High similarity detected! ({cosine_similarity_score:.2f}%). This text might be plagiarized."
            is_plagiarized_result = True
        elif cosine_similarity_score > 40 and prediction_label == 1:
            message = f"Moderate similarity ({cosine_similarity_score:.2f}%) and flagged by model. Review carefully."
            is_plagiarized_result = True
        elif cosine_similarity_score > 20:
            message = f"Some similarity detected ({cosine_similarity_score:.2f}%). Text might need review."

        found_sources = []
        if len(text_to_check.split()) > 15:
            found_sources = search_web_for_plagiarism(text_to_check)

        # Save to database
        save_to_db(user_id, text_to_check, is_plagiarized_result, cosine_similarity_score, message, found_sources)

        return jsonify({
            'is_plagiarized': is_plagiarized_result,
            'similarity_score': cosine_similarity_score,
            'message': message,
            'sources': found_sources
        })
    except Exception as e:
        print(f"Error during plagiarism check: {e}")
        return jsonify({'error': str(e), 'message': 'An error occurred during the check.'}), 500


@app.route('/get-history', methods=['POST'])
def get_history_route():
    try:
        user_id = request.json.get('user_id', 'anonymous')
        limit = request.json.get('limit', 10)

        history = get_history(user_id, limit)
        return jsonify({'history': history})
    except Exception as e:
        print(f"Error fetching history: {e}")
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True)