import librosa
import numpy as np

# Load audio file
y, sr = librosa.load('song.wav')

# Apply HPSS algorithm to separate vocals from accompaniment
y_harmonic, y_percussive = librosa.effects.hpss(y)

# Invert the separation to reconstruct the vocal component
vocals = y_harmonic - y_percussive

# Write the extracted vocals to a new audio file
librosa.output.write_wav('vocals.wav', vocals, sr)