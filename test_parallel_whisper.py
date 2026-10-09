import time
import os
import glob
from concurrent.futures import ProcessPoolExecutor
from faster_whisper import WhisperModel

base_dir = r"c:\Users\chand\Downloads\shl-hiring-assessment-2026\Dataset_Final\train"
files = glob.glob(os.path.join(base_dir, "*.wav"))[:8]

def transcribe_file(f):
    # Worker function initializes or uses model
    global worker_model
    if 'worker_model' not in globals():
        worker_model = WhisperModel("tiny.en", device="cpu", compute_type="int8", cpu_threads=2)
    segments, info = worker_model.transcribe(f, beam_size=1)
    text = " ".join([s.text for s in segments])
    return f, text, info.duration

if __name__ == '__main__':
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=4) as executor:
        results = list(executor.map(transcribe_file, files))
    t1 = time.time()
    print(f"Transcribed {len(results)} files in {t1 - t0:.2f} seconds! (Average {(t1 - t0)/len(results):.2f}s per file)")
    for f, text, dur in results[:2]:
        print(f"{os.path.basename(f)} ({dur:.1f}s): {text[:80]}...")
