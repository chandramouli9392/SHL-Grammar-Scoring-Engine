import os
import sys
import time
import subprocess
import pandas as pd

def log_header(title):
    print("\n" + "="*70)
    print(f"  {title}")
    print("="*70)

def main():
    root_dir = r"c:\Users\chand\Downloads\shl-hiring-assessment-2026"
    features_dir = os.path.join(root_dir, "features")
    os.makedirs(features_dir, exist_ok=True)
    
    # Step 1: Acoustic features check
    log_header("STEP 1: ACOUSTIC FEATURES EXTRACTION")
    ac_train = os.path.join(features_dir, "acoustic_features_train.csv")
    ac_test = os.path.join(features_dir, "acoustic_features_test.csv")
    
    if os.path.exists(ac_train) and os.path.exists(ac_test):
        df_actr = pd.read_csv(ac_train)
        df_acte = pd.read_csv(ac_test)
        if len(df_actr) >= 769 and len(df_acte) >= 216:
            print(f"[OK] Acoustic features already extracted: Train ({len(df_actr)}), Test ({len(df_acte)})")
        else:
            print(f"[INFO] Acoustic features partially extracted ({len(df_actr)} train, {len(df_acte)} test). Running extract_acoustic_features.py...")
            subprocess.run([sys.executable, "extract_acoustic_features.py"], cwd=root_dir, check=True)
    else:
        print("[INFO] Running extract_acoustic_features.py...")
        subprocess.run([sys.executable, "extract_acoustic_features.py"], cwd=root_dir, check=True)

    # Step 2: ASR Transcripts
    log_header("STEP 2: ASR SPEECH-TO-TEXT TRANSCRIPTION (faster-whisper)")
    tr_train = os.path.join(features_dir, "transcripts_train.csv")
    tr_test = os.path.join(features_dir, "transcripts_test.csv")
    
    needs_transcribe = True
    if os.path.exists(tr_train) and os.path.exists(tr_test):
        df_trtr = pd.read_csv(tr_train)
        df_trte = pd.read_csv(tr_test)
        if len(df_trtr) >= 769 and len(df_trte) >= 216:
            print(f"[OK] Transcripts already extracted: Train ({len(df_trtr)}), Test ({len(df_trte)})")
            needs_transcribe = False
            
    if needs_transcribe:
        print("[INFO] Running extract_transcripts.py with parallel workers...")
        subprocess.run([sys.executable, "extract_transcripts.py"], cwd=root_dir, check=True)

    # Step 3: Text & Linguistic features
    log_header("STEP 3: LINGUISTIC & GRAMMAR FEATURE EXTRACTION")
    tx_train = os.path.join(features_dir, "text_features_train.csv")
    tx_test = os.path.join(features_dir, "text_features_test.csv")
    
    needs_text_feat = True
    if os.path.exists(tx_train) and os.path.exists(tx_test):
        df_txtr = pd.read_csv(tx_train)
        df_txte = pd.read_csv(tx_test)
        if len(df_txtr) >= 769 and len(df_txte) >= 216:
            print(f"[OK] Text features already extracted: Train ({len(df_txtr)}), Test ({len(df_txte)})")
            needs_text_feat = False
            
    if needs_text_feat:
        print("[INFO] Running extract_text_features.py...")
        subprocess.run([sys.executable, "extract_text_features.py"], cwd=root_dir, check=True)

    # Step 4: Model Training, Ensembling, & Submission
    log_header("STEP 4: 5-FOLD MULTI-MODEL ENSEMBLE TRAINING & PREDICTION")
    print("[INFO] Running train_and_predict.py...")
    subprocess.run([sys.executable, "train_and_predict.py"], cwd=root_dir, check=True)

    log_header("PIPELINE COMPLETED SUCCESSFULLY!")
    print("Generated Submission Files:")
    for fn in ["submission.csv", "submission_sample_format.csv"]:
        fp = os.path.join(root_dir, fn)
        if os.path.exists(fp):
            df_sub = pd.read_csv(fp)
            print(f"  - {fn}: {len(df_sub)} rows, columns: {list(df_sub.columns)}")

if __name__ == '__main__':
    main()
