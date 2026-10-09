import os
import sys
import time
import glob
import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor
from faster_whisper import WhisperModel

base_dir = r"c:\Users\chand\Downloads\shl-hiring-assessment-2026\Dataset_Final"
features_dir = r"c:\Users\chand\Downloads\shl-hiring-assessment-2026\features"
os.makedirs(features_dir, exist_ok=True)

# Global model variable in child processes
worker_model = None

def init_worker():
    global worker_model
    worker_model = WhisperModel("tiny.en", device="cpu", compute_type="int8", cpu_threads=2)

def transcribe_single_file(args):
    global worker_model
    file_path, filename, label = args
    if worker_model is None:
        init_worker()
        
    try:
        segments, info = worker_model.transcribe(file_path, beam_size=1)
        segments_list = list(segments)
        full_text = " ".join([s.text.strip() for s in segments_list]).strip()
        
        duration = float(info.duration)
        words = full_text.split()
        word_count = len(words)
        wpm = (word_count / (duration / 60.0)) if duration > 0 else 0.0
        wps = (word_count / duration) if duration > 0 else 0.0
        
        segment_durations = []
        pause_between_segments = []
        logprobs = []
        compression_ratios = []
        no_speech_probs = []
        
        prev_end = 0.0
        for s in segments_list:
            seg_dur = max(0.0, s.end - s.start)
            segment_durations.append(seg_dur)
            if s.start > prev_end and prev_end > 0.0:
                pause_between_segments.append(s.start - prev_end)
            prev_end = s.end
            
            if hasattr(s, 'avg_logprob') and s.avg_logprob is not None:
                logprobs.append(s.avg_logprob)
            if hasattr(s, 'compression_ratio') and s.compression_ratio is not None:
                compression_ratios.append(s.compression_ratio)
            if hasattr(s, 'no_speech_prob') and s.no_speech_prob is not None:
                no_speech_probs.append(s.no_speech_prob)
                
        return {
            'filename': filename,
            'label': label,
            'transcript': full_text,
            'duration': duration,
            'word_count': word_count,
            'words_per_minute': wpm,
            'words_per_second': wps,
            'segment_count': len(segments_list),
            'avg_seg_duration': float(np.mean(segment_durations)) if segment_durations else 0.0,
            'std_seg_duration': float(np.std(segment_durations)) if segment_durations else 0.0,
            'avg_inter_seg_pause': float(np.mean(pause_between_segments)) if pause_between_segments else 0.0,
            'max_inter_seg_pause': float(np.max(pause_between_segments)) if pause_between_segments else 0.0,
            'whisper_avg_logprob': float(np.mean(logprobs)) if logprobs else 0.0,
            'whisper_compression_ratio': float(np.mean(compression_ratios)) if compression_ratios else 0.0,
            'whisper_no_speech_prob': float(np.mean(no_speech_probs)) if no_speech_probs else 0.0,
        }
    except Exception as e:
        print(f"Error transcribing {filename}: {e}", flush=True)
        return {
            'filename': filename,
            'label': label,
            'transcript': "",
            'duration': 0.0,
            'word_count': 0,
            'words_per_minute': 0.0,
            'words_per_second': 0.0,
            'segment_count': 0,
            'avg_seg_duration': 0.0,
            'std_seg_duration': 0.0,
            'avg_inter_seg_pause': 0.0,
            'max_inter_seg_pause': 0.0,
            'whisper_avg_logprob': 0.0,
            'whisper_compression_ratio': 0.0,
            'whisper_no_speech_prob': 0.0,
        }

def process_transcriptions(file_args, output_path, desc="", num_workers=4):
    existing_filenames = set()
    existing_records = []
    if os.path.exists(output_path):
        try:
            df_existing = pd.read_csv(output_path)
            existing_filenames = set(df_existing['filename'])
            existing_records = df_existing.to_dict('records')
            print(f"[{desc}] Found {len(existing_filenames)} already transcribed files in {output_path}", flush=True)
        except Exception as e:
            print(f"Could not load existing file {output_path}: {e}", flush=True)

    to_process = [arg for arg in file_args if arg[1] not in existing_filenames]
    print(f"[{desc}] Total {len(file_args)}, remaining to transcribe: {len(to_process)}", flush=True)

    if not to_process:
        print(f"[{desc}] All files already transcribed!", flush=True)
        return pd.read_csv(output_path)

    all_records = list(existing_records)
    t_start = time.time()
    batch_size = 15
    
    with ProcessPoolExecutor(max_workers=num_workers, initializer=init_worker) as executor:
        for i in range(0, len(to_process), batch_size):
            batch_args = to_process[i:i + batch_size]
            batch_res = list(executor.map(transcribe_single_file, batch_args))
            all_records.extend(batch_res)
            # Save checkpoint
            pd.DataFrame(all_records).to_csv(output_path, index=False)
            elapsed = time.time() - t_start
            done_count = len(all_records) - len(existing_records)
            rate = done_count / (elapsed + 1e-5)
            eta = (len(to_process) - done_count) / (rate + 1e-5)
            print(f"[{desc}] Checkpoint: {len(all_records)}/{len(file_args)} ({done_count}/{len(to_process)} done in {elapsed:.1f}s, {rate:.2f} files/s, ETA: {eta/60:.1f}m)", flush=True)

    df_final = pd.DataFrame(all_records)
    df_final.to_csv(output_path, index=False)
    print(f"[{desc}] Completed in {time.time() - t_start:.1f}s! Saved to {output_path}", flush=True)
    return df_final

if __name__ == '__main__':
    train_csv = pd.read_csv(os.path.join(base_dir, "train.csv"))
    test_csv = pd.read_csv(os.path.join(base_dir, "test.csv"))
    
    train_args = [(os.path.join(base_dir, "train", row['filename']), row['filename'], row['label']) for _, row in train_csv.iterrows()]
    test_args = [(os.path.join(base_dir, "test", row['filename']), row['filename'], row['label']) for _, row in test_csv.iterrows()]
    
    num_workers = 4
    
    print("\n" + "="*50, flush=True)
    print("STEP 2: ASR SPEECH-TO-TEXT EXTRACTION (faster-whisper tiny.en)", flush=True)
    print("="*50, flush=True)
    
    # Process test first (216 files)
    print("\n--- Transcribing Test Set (216 files) ---", flush=True)
    df_test_trans = process_transcriptions(test_args, os.path.join(features_dir, "transcripts_test.csv"), "Test", num_workers=num_workers)
    
    # Process train (769 files)
    print("\n--- Transcribing Train Set (769 files) ---", flush=True)
    df_train_trans = process_transcriptions(train_args, os.path.join(features_dir, "transcripts_train.csv"), "Train", num_workers=num_workers)
