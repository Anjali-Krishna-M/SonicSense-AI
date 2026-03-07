from flask import Flask, render_template, request, session, redirect, url_for, send_from_directory, flash
import librosa
import numpy as np
import joblib
import os
import uuid
import sqlite3
from datetime import datetime
from werkzeug.security import generate_password_hash, check_password_hash
import mimetypes
mimetypes.add_type('text/css', '.css')


app = Flask(__name__)
app.secret_key = "super_secret_mca_key_viva_2026"

UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# --- INITIALIZE DATABASE ---
def init_db():
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    # Create a table for users if it doesn't exist
    c.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL
        )
    ''')
    conn.commit()
    conn.close()

# Run database init on startup
init_db()

print("⏳ Loading AI Models...")
try:
    model = joblib.load('models/music_genre_model.pkl')
    scaler = joblib.load('models/scaler.pkl')
    encoder = joblib.load('models/encoder.pkl')
except:
    print("❌ ERROR: Models not found. Run train_model.py!")

# --- REAL SONG DATABASE ---
REAL_SONGS = {
    'blues': [{'title': 'The Thrill Is Gone', 'artist': 'B.B. King'}, {'title': 'Pride and Joy', 'artist': 'Stevie Ray Vaughan'}, {'title': 'Cross Road Blues', 'artist': 'Robert Johnson'}, {'title': 'Sweet Home Chicago', 'artist': 'Buddy Guy'}, {'title': 'Hoochie Coochie Man', 'artist': 'Muddy Waters'}],
    'classical': [{'title': 'Nocturne No. 2', 'artist': 'Chopin'}, {'title': 'Moonlight Sonata', 'artist': 'Beethoven'}, {'title': 'Four Seasons', 'artist': 'Vivaldi'}, {'title': 'Clair de Lune', 'artist': 'Debussy'}, {'title': 'Ride of the Valkyries', 'artist': 'Wagner'}],
    'country': [{'title': 'Take Me Home, Country Roads', 'artist': 'John Denver'}, {'title': 'Jolene', 'artist': 'Dolly Parton'}, {'title': 'Tennessee Whiskey', 'artist': 'Chris Stapleton'}, {'title': 'Friends in Low Places', 'artist': 'Garth Brooks'}, {'title': 'Ring of Fire', 'artist': 'Johnny Cash'}],
    'disco': [{'title': 'Stayin\' Alive', 'artist': 'Bee Gees'}, {'title': 'Dancing Queen', 'artist': 'ABBA'}, {'title': 'I Will Survive', 'artist': 'Gloria Gaynor'}, {'title': 'Le Freak', 'artist': 'Chic'}, {'title': 'September', 'artist': 'Earth, Wind & Fire'}],
    'hiphop': [{'title': 'Lose Yourself', 'artist': 'Eminem'}, {'title': 'Juicy', 'artist': 'The Notorious B.I.G.'}, {'title': 'Sicko Mode', 'artist': 'Travis Scott'}, {'title': 'God\'s Plan', 'artist': 'Drake'}, {'title': 'N.Y. State of Mind', 'artist': 'Nas'}],
    'jazz': [{'title': 'Take Five', 'artist': 'Dave Brubeck'}, {'title': 'So What', 'artist': 'Miles Davis'}, {'title': 'Fly Me To The Moon', 'artist': 'Frank Sinatra'}, {'title': 'What A Wonderful World', 'artist': 'Louis Armstrong'}, {'title': 'Strange Fruit', 'artist': 'Billie Holiday'}],
    'metal': [{'title': 'Master of Puppets', 'artist': 'Metallica'}, {'title': 'Paranoid', 'artist': 'Black Sabbath'}, {'title': 'Chop Suey!', 'artist': 'System of a Down'}, {'title': 'Enter Sandman', 'artist': 'Metallica'}, {'title': 'Run to the Hills', 'artist': 'Iron Maiden'}],
    'pop': [{'title': 'Blinding Lights', 'artist': 'The Weeknd'}, {'title': 'Shape of You', 'artist': 'Ed Sheeran'}, {'title': 'Bad Guy', 'artist': 'Billie Eilish'}, {'title': 'Levitating', 'artist': 'Dua Lipa'}, {'title': 'Thriller', 'artist': 'Michael Jackson'}],
    'reggae': [{'title': 'Three Little Birds', 'artist': 'Bob Marley'}, {'title': 'Red Red Wine', 'artist': 'UB40'}, {'title': 'Welcome to Jamrock', 'artist': 'Damian Marley'}, {'title': 'Could You Be Loved', 'artist': 'Bob Marley'}, {'title': 'Boombastic', 'artist': 'Shaggy'}],
    'rock': [{'title': 'Bohemian Rhapsody', 'artist': 'Queen'}, {'title': 'Smells Like Teen Spirit', 'artist': 'Nirvana'}, {'title': 'Hotel California', 'artist': 'Eagles'}, {'title': 'Sweet Child O\' Mine', 'artist': 'Guns N\' Roses'}, {'title': 'Back In Black', 'artist': 'AC/DC'}]
}

genre_details = {
    'blues': {'mood': 'Melancholy & Soulful 🎷', 'desc': 'Deeply emotional, rooted in struggle.'},
    'classical': {'mood': 'Calm & Intellectual 🎻', 'desc': 'Complex and structured. Evokes focus.'},
    'country': {'mood': 'Nostalgic & Warm 🤠', 'desc': 'Storytelling-based music evoking home.'},
    'disco': {'mood': 'Energetic & Party 🕺', 'desc': 'High-energy dance music designed to move.'},
    'hiphop': {'mood': 'Confident & Rhythmic 🎤', 'desc': 'Beat-driven and lyrical urban culture.'},
    'jazz': {'mood': 'Sophisticated & Relaxed 🎹', 'desc': 'Improvisational, complex, and smoky.'},
    'metal': {'mood': 'Aggressive & Intense 🎸', 'desc': 'High distortion, raw energy, and power.'},
    'pop': {'mood': 'Happy & Upbeat 🍭', 'desc': 'Catchy, melodic, and structured positivity.'},
    'reggae': {'mood': 'Chill & Groovy 🇯🇲', 'desc': 'Laid-back offbeat rhythms for peace.'},
    'rock': {'mood': 'Rebellious & Strong 🤘', 'desc': 'Driving rhythms representing raw emotion.'}
}

def extract_features(file_path):
    y, sr = librosa.load(file_path, mono=True, duration=30)
    chroma_stft = np.mean(librosa.feature.chroma_stft(y=y, sr=sr))
    rms = np.mean(librosa.feature.rms(y=y))
    spec_cent = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
    spec_bw = np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr))
    rolloff = np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr))
    zcr = np.mean(librosa.feature.zero_crossing_rate(y))
    mfcc = librosa.feature.mfcc(y=y, sr=sr)
    features = [chroma_stft, rms, spec_cent, spec_bw, rolloff, zcr]
    for e in mfcc: features.append(np.mean(e))
    chart_data = [float(rms)*100, float(spec_cent)/50, float(rolloff)/100, float(zcr)*500, 120]
    return np.array(features).reshape(1, -1), chart_data

# --- AUTHENTICATION ROUTES ---
@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        hashed_pw = generate_password_hash(password) # Encrypt password
        
        try:
            conn = sqlite3.connect('users.db')
            c = conn.cursor()
            c.execute("INSERT INTO users (username, password) VALUES (?, ?)", (username, hashed_pw))
            conn.commit()
            conn.close()
            flash('Registration successful! Please login.', 'success')
            return redirect(url_for('login'))
        except sqlite3.IntegrityError:
            flash('Username already exists. Please choose another.', 'error')
            
    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        
        conn = sqlite3.connect('users.db')
        c = conn.cursor()
        c.execute("SELECT * FROM users WHERE username=?", (username,))
        user = c.fetchone()
        conn.close()
        
        if user and check_password_hash(user[2], password):
            session['user'] = username # Create session
            return redirect(url_for('index'))
        else:
            flash('Invalid username or password', 'error')
            
    return render_template('login.html')

@app.route('/logout')
def logout():
    session.pop('user', None) # Clear session
    session.pop('history', None)
    session.pop('result', None)
    return redirect(url_for('login'))

# --- CORE ROUTES ---
@app.route('/')
def index(): 
    if 'user' not in session: return redirect(url_for('login'))
    return render_template('index.html')

@app.route('/about')
def about(): 
    return render_template('about.html')

@app.route('/analytics')
def analytics(): 
    if 'user' not in session: return redirect(url_for('login'))
    return render_template('analytics.html')

@app.route('/history')
def history(): 
    if 'user' not in session: return redirect(url_for('login'))
    return render_template('history.html', history=session.get('history', []))

@app.route('/dashboard')
def dashboard():
    if 'user' not in session: return redirect(url_for('login'))
    if 'result' not in session: return redirect(url_for('index'))
    return render_template('dashboard.html', data=session['result'])

@app.route('/recommendations')
def recommendations():
    if 'user' not in session: return redirect(url_for('login'))
    if 'result' not in session: return redirect(url_for('index'))
    return render_template('recommendations.html', data=session['result'])

@app.route('/analyze', methods=['POST'])
def analyze():
    if 'user' not in session: return redirect(url_for('login'))
    if 'file' not in request.files: return redirect(url_for('index'))
    file = request.files['file']
    if file.filename == '': return redirect(url_for('index'))
    
    original_name = file.filename
    unique_filename = str(uuid.uuid4()) + ".wav"
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
    file.save(file_path)

    try:
        feat, chart_data = extract_features(file_path)
        scaled_feat = scaler.transform(feat)
        pred_idx = model.predict(scaled_feat)[0]
        genre = encoder.inverse_transform([pred_idx])[0]
        confidence = np.max(model.predict_proba(scaled_feat)) * 100
        recs = REAL_SONGS.get(genre, [])
        
        result_data = {
            'genre': genre.upper(), 'mood': genre_details.get(genre, {}).get('mood', 'Unknown'),
            'desc': genre_details.get(genre, {}).get('desc', ''), 'confidence': f"{confidence:.1f}",
            'chart_data': chart_data, 'recommendations': recs,
            'uploaded_file': unique_filename, 'original_name': original_name
        }
        session['result'] = result_data

        if 'history' not in session: session['history'] = []
        session['history'].insert(0, {'name': original_name, 'genre': genre.upper(), 'date': datetime.now().strftime("%H:%M")})
        session['history'] = session['history'][:10]
        return redirect(url_for('dashboard'))
    except Exception as e:
        print(f"Error: {e}")
        return redirect(url_for('index'))

@app.route('/play_uploaded/<filename>')
def play_uploaded(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True)