import os
import time
import numpy as np
import pandas as pd
import soundfile as sf
import librosa
from scipy.stats import skew, kurtosis
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm

def extract_single_audio_features(args):
    file_path, filename, label = args
    try:
        y, sr = sf.read(file_path)
        if len(y.shape) > 1:
            y = np.mean(y, axis=1)
        
        # Audio length & basic stats
        total_samples = len(y)
        duration = total_samples / sr
        
        # Silence & Energy calculation
        # Frame-wise RMS
        frame_length = int(sr * 0.05) # 50ms frames
        hop_length = int(sr * 0.025)   # 25ms hop
        rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]
        
        rms_threshold = np.max(rms) * 0.05 if np.max(rms) > 0 else 0.001
        is_speech = rms > rms_threshold
        speech_frames = np.sum(is_speech)
        silence_frames = len(rms) - speech_frames
        speech_duration = (speech_frames * hop_length) / sr
        silence_duration = (silence_frames * hop_length) / sr
        speech_ratio = speech_frames / (len(rms) + 1e-8)
        
        # Count pauses (> 250ms silence)
        pause_counts = 0
        pause_durations = []
        cur_pause = 0
        for s in is_speech:
            if not s:
                cur_pause += hop_length / sr
            else:
                if cur_pause >= 0.25:
                    pause_counts += 1
                    pause_durations.append(cur_pause)
                cur_pause = 0
        if cur_pause >= 0.25:
            pause_counts += 1
            pause_durations.append(cur_pause)
            
        mean_pause_dur = np.mean(pause_durations) if pause_durations else 0.0
        max_pause_dur = np.max(pause_durations) if pause_durations else 0.0
        
        # RMS stats
        rms_mean = float(np.mean(rms))
        rms_std = float(np.std(rms))
        rms_skew = float(skew(rms)) if len(rms) > 2 else 0.0
        rms_kurt = float(kurtosis(rms)) if len(rms) > 2 else 0.0
        
        # ZCR
        zcr = librosa.feature.zero_crossing_rate(y=y, frame_length=frame_length, hop_length=hop_length)[0]
        zcr_mean = float(np.mean(zcr))
        zcr_std = float(np.std(zcr))
        zcr_max = float(np.max(zcr))
        
        # Spectral Centroid, Bandwidth, Rolloff, Flatness
        sc = librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=hop_length)[0]
        sb = librosa.feature.spectral_bandwidth(y=y, sr=sr, hop_length=hop_length)[0]
        s_roll85 = librosa.feature.spectral_rolloff(y=y, sr=sr, roll_percent=0.85, hop_length=hop_length)[0]
        s_flat = librosa.feature.spectral_flatness(y=y, hop_length=hop_length)[0]
        
        # Spectral Contrast
        s_contrast = librosa.feature.spectral_contrast(y=y, sr=sr, hop_length=hop_length)
        contrast_means = [float(np.mean(s_contrast[i])) for i in range(s_contrast.shape[0])]
        
        # MFCC (20 coefficients)
        n_mfcc = 20
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc, hop_length=hop_length)
        delta_mfcc = librosa.feature.delta(mfcc)
        delta2_mfcc = librosa.feature.delta(mfcc, order=2)
        
        # Pitch estimation using librosa.yin (reliable and fast on downsampled speech)
        # Downsample to 8kHz for fast pitch tracking
        y_8k = librosa.resample(y, orig_sr=sr, target_sr=8000)
        f0 = librosa.yin(y_8k, fmin=50, fmax=400, sr=8000, hop_length=200)
        valid_f0 = f0[(f0 >= 55) & (f0 <= 390)]
        voicing_fraction = len(valid_f0) / (len(f0) + 1e-8)
        if len(valid_f0) > 0:
            f0_mean = float(np.mean(valid_f0))
            f0_std = float(np.std(valid_f0))
            f0_min = float(np.min(valid_f0))
            f0_max = float(np.max(valid_f0))
            f0_range = f0_max - f0_min
        else:
            f0_mean, f0_std, f0_min, f0_max, f0_range = 0.0, 0.0, 0.0, 0.0, 0.0
            
        feature_dict = {
            'filename': filename,
            'label': label,
            'duration': duration,
            'speech_duration': speech_duration,
            'silence_duration': silence_duration,
            'speech_ratio': speech_ratio,
            'pause_count': pause_counts,
            'pause_rate': pause_counts / (duration + 1e-8),
            'mean_pause_dur': mean_pause_dur,
            'max_pause_dur': max_pause_dur,
            'rms_mean': rms_mean,
            'rms_std': rms_std,
            'rms_skew': rms_skew,
            'rms_kurt': rms_kurt,
            'zcr_mean': zcr_mean,
            'zcr_std': zcr_std,
            'zcr_max': zcr_max,
            'sc_mean': float(np.mean(sc)),
            'sc_std': float(np.std(sc)),
            'sc_skew': float(skew(sc)),
            'sb_mean': float(np.mean(sb)),
            'sb_std': float(np.std(sb)),
            'roll85_mean': float(np.mean(s_roll85)),
            'roll85_std': float(np.std(s_roll85)),
            'flatness_mean': float(np.mean(s_flat)),
            'flatness_std': float(np.std(s_flat)),
            'f0_mean': f0_mean,
            'f0_std': f0_std,
            'f0_min': f0_min,
            'f0_max': f0_max,
            'f0_range': f0_range,
            'voicing_fraction': voicing_fraction,
        }
        
        # Add contrast bands
        for i, cm in enumerate(contrast_means):
            feature_dict[f'contrast_band_{i}_mean'] = cm
            
        # Add MFCC stats
        for i in range(n_mfcc):
            feature_dict[f'mfcc_{i}_mean'] = float(np.mean(mfcc[i]))
            feature_dict[f'mfcc_{i}_std'] = float(np.std(mfcc[i]))
            feature_dict[f'mfcc_{i}_skew'] = float(skew(mfcc[i]))
            feature_dict[f'mfcc_delta_{i}_mean'] = float(np.mean(delta_mfcc[i]))
            feature_dict[f'mfcc_delta_{i}_std'] = float(np.std(delta_mfcc[i]))
            feature_dict[f'mfcc_delta2_{i}_mean'] = float(np.mean(delta2_mfcc[i]))
            feature_dict[f'mfcc_delta2_{i}_std'] = float(np.std(delta2_mfcc[i]))
            
        return feature_dict
    except Exception as e:
        print(f"Error extracting features for {filename}: {e}")
        return None

if __name__ == '__main__':
    base_dir = r"c:\Users\chand\Downloads\shl-hiring-assessment-2026\Dataset_Final"
    features_dir = r"c:\Users\chand\Downloads\shl-hiring-assessment-2026\features"
    os.makedirs(features_dir, exist_ok=True)
    
    train_csv = pd.read_csv(os.path.join(base_dir, "train.csv"))
    test_csv = pd.read_csv(os.path.join(base_dir, "test.csv"))
    
    train_args = [(os.path.join(base_dir, "train", row['filename']), row['filename'], row['label']) for _, row in train_csv.iterrows()]
    test_args = [(os.path.join(base_dir, "test", row['filename']), row['filename'], row['label']) for _, row in test_csv.iterrows()]
    
    print("Testing single file extraction...")
    t0 = time.time()
    res = extract_single_audio_features(train_args[0])
    print(f"Single file extracted in {time.time() - t0:.2f}s! Extracted {len(res)} features.")
    
    num_workers = min(8, os.cpu_count() or 4)
    print(f"Using {num_workers} parallel workers")

    # Helper function to process in batches with checkpointing
    def extract_and_save(file_args, output_path, desc=""):
        existing_filenames = set()
        existing_records = []
        if os.path.exists(output_path):
            try:
                df_existing = pd.read_csv(output_path)
                existing_filenames = set(df_existing['filename'])
                existing_records = df_existing.to_dict('records')
                print(f"Found {len(existing_filenames)} already processed files in {output_path}")
            except Exception as e:
                print(f"Could not load existing file {output_path}: {e}")

        to_process = [arg for arg in file_args if arg[1] not in existing_filenames]
        print(f"{desc}: Total {len(file_args)}, remaining to process: {len(to_process)}")

        if not to_process:
            print(f"All files already processed for {desc}!")
            return pd.read_csv(output_path)

        all_records = list(existing_records)
        t_start = time.time()
        batch_size = 50
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            for i in range(0, len(to_process), batch_size):
                batch_args = to_process[i:i + batch_size]
                batch_res = list(executor.map(extract_single_audio_features, batch_args))
                valid_res = [r for r in batch_res if r is not None]
                all_records.extend(valid_res)
                # Checkpoint save
                pd.DataFrame(all_records).to_csv(output_path, index=False)
                elapsed = time.time() - t_start
                done_count = len(all_records) - len(existing_records)
                rate = done_count / (elapsed + 1e-5)
                eta = (len(to_process) - done_count) / (rate + 1e-5)
                print(f"[{desc}] Saved checkpoint: {len(all_records)}/{len(file_args)} processed ({done_count}/{len(to_process)} done in {elapsed:.1f}s, ETA: {eta:.1f}s)")

        df_final = pd.DataFrame(all_records)
        df_final.to_csv(output_path, index=False)
        print(f"[{desc}] Complete! Saved {len(df_final)} rows to {output_path}")
        return df_final

    print("\n--- Extracting Train Acoustic Features ---")
    df_train_feat = extract_and_save(train_args, os.path.join(features_dir, "acoustic_features_train.csv"), "Train")

    print("\n--- Extracting Test Acoustic Features ---")
    df_test_feat = extract_and_save(test_args, os.path.join(features_dir, "acoustic_features_test.csv"), "Test")

