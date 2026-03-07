
# 🎵 SonicSense AI: Intelligent Music Genre & Mood Classifier

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![Flask](https://img.shields.io/badge/Flask-2.0-green)
![TensorFlow](https://img.shields.io/badge/ML-Scikit--Learn-orange)

**SonicSense AI** is an advanced audio analysis system designed to classify music genres, predict emotional mood, and visualize the "Audio DNA" of any track using Digital Signal Processing (DSP) and Machine Learning.

## 🚀 Features
* **Genre Classification:** accurately predicts 10 genres (Rock, Pop, Jazz, etc.) using a Random Forest ensemble trained on the GTZAN dataset.
* **Mood Detection:** Maps spectral features to emotional states (e.g., "Happy & Upbeat", "Aggressive & Intense").
* **Audio DNA Visualizer:** A dynamic Radar Chart visualizing Energy, Timbre, Roughness, and Rhythm.
* **Smart Recommendations:** Suggests 5 real-world hit songs (with playback links) based on the track's audio signature.
* **Session History:** Tracks your analysis session for comparison.

## 🛠️ Tech Stack
* **Backend:** Python, Flask
* **Audio Processing:** Librosa, NumPy, Pandas
* **Machine Learning:** Scikit-learn (Random Forest Classifier)
* **Frontend:** HTML5, CSS3 (Glassmorphism UI), JavaScript (Chart.js)

---


### 1. Clone the Repository
```bash
git clone [https://github.com/YOUR_USERNAME/SonicSense-AI.git](https://github.com/YOUR_USERNAME/SonicSense-AI.git)
cd SonicSense-AI

```

### 2. Set Up Virtual Environment

```bash
# Windows
python -m venv venv
.\venv\Scripts\activate

# Mac/Linux
python3 -m venv venv
source venv/bin/activate

```

### 3. Install Dependencies

```bash
pip install -r requirements.txt

```

### 4. Download the Dataset

Due to size limits, the dataset is not included in this repo.

1. Download the **GTZAN Dataset** from [Kaggle](https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification).
2. Extract it.
3. Copy the `genres_original` folder into a new folder named `dataset` in this project.
* Structure: `Music_Genre_Project/dataset/genres_original/blues/...`



---

## 🏃‍♂️ How to Run

1. **Extract Features (First time only)**
Process the raw audio files into a CSV dataset.
```bash
python extract_features.py

```


2. **Train the Model (First time only)**
Train the Machine Learning model.
```bash
python train_model.py

```


3. **Start the Server**
```bash
python app.py

```


4. Open your browser and go to: `http://127.0.0.1:5000`

---

## 📄 License

This project is for educational purposes (MCA Final Year Project).


