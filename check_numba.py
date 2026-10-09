import time
import soundfile as sf
import librosa
import numpy as np

test_audio = r"c:\Users\chand\Downloads\shl-hiring-assessment-2026\Dataset_Final\train\audio_192.wav"
y, sr = sf.read(test_audio)
if len(y.shape) > 1:
    y = np.mean(y, axis=1)

# First call (JIT already compiled if warm, but let's test)
t0 = time.time()
mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
print(f"Call 1 MFCC took {time.time() - t0:.4f}s")

t1 = time.time()
mfcc2 = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
print(f"Call 2 MFCC took {time.time() - t1:.4f}s")

t2 = time.time()
sc = librosa.feature.spectral_centroid(y=y, sr=sr)
print(f"Call 1 Centroid took {time.time() - t2:.4f}s")

t3 = time.time()
sc2 = librosa.feature.spectral_centroid(y=y, sr=sr)
print(f"Call 2 Centroid took {time.time() - t3:.4f}s")
