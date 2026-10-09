import time
import os
import soundfile as sf
import librosa
import numpy as np
from faster_whisper import WhisperModel

test_audio = r"c:\Users\chand\Downloads\shl-hiring-assessment-2026\Dataset_Final\train\audio_192.wav"

# Benchmark acoustic features extraction
t0 = time.time()
y, sr = sf.read(test_audio)
if len(y.shape) > 1:
    y = np.mean(y, axis=1)

# Extract MFCC, chroma, spectral, pitch
mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
delta_mfcc = librosa.feature.delta(mfcc)
delta2_mfcc = librosa.feature.delta(mfcc, order=2)
centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
zcr = librosa.feature.zero_crossing_rate(y)
rms = librosa.feature.rms(y=y)
t1 = time.time()
print(f"Acoustic features extraction took {t1 - t0:.2f} seconds")

# Benchmark tiny.en faster-whisper
t2 = time.time()
model_tiny = WhisperModel("tiny.en", device="cpu", compute_type="int8", cpu_threads=4)
t3 = time.time()
print(f"Tiny.en model load took {t3 - t2:.2f} seconds")
segments, info = model_tiny.transcribe(test_audio, beam_size=1)
text = " ".join([s.text for s in segments])
t4 = time.time()
print(f"Tiny.en transcription took {t4 - t3:.2f} seconds")
print("Tiny transcript preview:", text[:150])
