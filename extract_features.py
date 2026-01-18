import librosa
import pandas as pd
import numpy as np
import os
import csv

# We are extracting these audio features:
# 1. Chroma Stft (Pitch)
# 2. RMS (Energy)
# 3. Spectral Centroid (Brightness)
# 4. Spectral Bandwidth (Range)
# 5. Rolloff (Shape)
# 6. Zero Crossing Rate (Percussion)
# 7. MFCCs (Timbre - very important!)

header = 'filename chroma_stft rms spectral_centroid spectral_bandwidth rolloff zero_crossing_rate'
for i in range(1, 21):
    header += f' mfcc{i}'
header += ' label'
header = header.split()

file = open('data.csv', 'w', newline='')
with file:
    writer = csv.writer(file)
    writer.writerow(header)

genres = 'blues classical country disco hiphop jazz metal pop reggae rock'.split()

print("🚀 Starting Feature Extraction... (This might take 5-10 minutes)")

for g in genres:
    for filename in os.listdir(f'./dataset/genres_original/{g}'):
        songname = f'./dataset/genres_original/{g}/{filename}'
        
        # Load audio file
        try:
            y, sr = librosa.load(songname, mono=True, duration=30)
            
            # Extract Features
            chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr)
            rms = librosa.feature.rms(y=y)
            spec_cent = librosa.feature.spectral_centroid(y=y, sr=sr)
            spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)
            rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
            zcr = librosa.feature.zero_crossing_rate(y)
            mfcc = librosa.feature.mfcc(y=y, sr=sr)
            
            # Calculate Mean of features
            to_append = f'{filename} {np.mean(chroma_stft)} {np.mean(rms)} {np.mean(spec_cent)} {np.mean(spec_bw)} {np.mean(rolloff)} {np.mean(zcr)}'    
            for e in mfcc:
                to_append += f' {np.mean(e)}'
            to_append += f' {g}'
            
            file = open('data.csv', 'a', newline='')
            with file:
                writer = csv.writer(file)
                writer.writerow(to_append.split())
        except Exception as e:
            print(f"Error processing {filename}: {e}")

    print(f"✅ Processed genre: {g}")

print("🎉 DONE! Features saved to data.csv")