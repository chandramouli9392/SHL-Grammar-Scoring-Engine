import time
import os
import soundfile as sf
from faster_whisper import WhisperModel

test_audio = r"c:\Users\chand\Downloads\shl-hiring-assessment-2026\Dataset_Final\train\audio_192.wav"
print(f"Testing faster-whisper on {test_audio}")

t0 = time.time()
# Use int8 compute type on CPU for fastest execution
model = WhisperModel("base.en", device="cpu", compute_type="int8")
t1 = time.time()
print(f"Model loaded in {t1 - t0:.2f} seconds")

segments, info = model.transcribe(test_audio, beam_size=1)
text = " ".join([segment.text for segment in segments])
t2 = time.time()
print(f"Transcription took {t2 - t1:.2f} seconds")
print(f"Detected language: {info.language} with prob {info.language_probability:.2f}")
print("Transcript preview:", text[:200])
