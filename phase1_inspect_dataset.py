import os
import glob
import numpy as np
import pandas as pd
import soundfile as sf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

base_dir = r"c:\Users\chand\Downloads\shl-hiring-assessment-2026\Dataset_Final"
output_dir = r"c:\Users\chand\Downloads\shl-hiring-assessment-2026\eda_reports"
os.makedirs(output_dir, exist_ok=True)

train_csv_path = os.path.join(base_dir, "train.csv")
test_csv_path = os.path.join(base_dir, "test.csv")
sample_sub_path = os.path.join(base_dir, "sample_submission.csv")
train_audio_dir = os.path.join(base_dir, "train")
test_audio_dir = os.path.join(base_dir, "test")

train_df = pd.read_csv(train_csv_path)
test_df = pd.read_csv(test_csv_path)
sample_sub_df = pd.read_csv(sample_sub_path)

print("="*60)
print("PHASE 1: COMPLETE DATASET INSPECTION")
print("="*60)

print(f"Train CSV rows: {len(train_df)}, columns: {train_df.columns.tolist()}")
print(f"Test CSV rows: {len(test_df)}, columns: {test_df.columns.tolist()}")
print(f"Sample Sub rows: {len(sample_sub_df)}, columns: {sample_sub_df.columns.tolist()}")

# First 5 rows of each
print("\nTrain Head:\n", train_df.head())
print("\nTest Head:\n", test_df.head())
print("\nSample Sub Head:\n", sample_sub_df.head())

# Target stats
labels = train_df['label']
stats = {
    'count': len(labels),
    'min': float(labels.min()),
    'max': float(labels.max()),
    'mean': float(labels.mean()),
    'median': float(labels.median()),
    'std': float(labels.std()),
    'q25': float(labels.quantile(0.25)),
    'q75': float(labels.quantile(0.75)),
}
print("\nTarget ('label') Statistics:")
for k, v in stats.items():
    print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

print("Target value counts:")
print(train_df['label'].value_counts().sort_index())

# Check missing and duplicates
print("\nMissing values in train:", train_df.isnull().sum().to_dict())
print("Missing values in test:", test_df.isnull().sum().to_dict())
print("Duplicate filenames in train:", train_df['filename'].duplicated().sum())
print("Duplicate filenames in test:", test_df['filename'].duplicated().sum())

# Audio file inspection
print("\nInspecting Audio Files...")
def inspect_audio_folder(folder_path, df, name=""):
    records = []
    corrupted = []
    for idx, row in tqdm(df.iterrows(), total=len(df), desc=f"Inspecting {name}"):
        fn = row['filename']
        fp = os.path.join(folder_path, fn)
        if not os.path.exists(fp):
            corrupted.append((fn, "File Not Found"))
            continue
        try:
            info = sf.info(fp)
            records.append({
                'filename': fn,
                'samplerate': info.samplerate,
                'channels': info.channels,
                'duration': info.duration,
                'frames': info.frames,
                'format': info.format,
                'subtype': info.subtype
            })
        except Exception as e:
            corrupted.append((fn, str(e)))
    return pd.DataFrame(records), corrupted

train_audio_df, train_corrupted = inspect_audio_folder(train_audio_dir, train_df, "Train")
test_audio_df, test_corrupted = inspect_audio_folder(test_audio_dir, test_df, "Test")

print(f"\nTrain audio inspected: {len(train_audio_df)} valid, {len(train_corrupted)} corrupted")
if train_corrupted:
    print("Corrupted train files:", train_corrupted)

print(f"Test audio inspected: {len(test_audio_df)} valid, {len(test_corrupted)} corrupted")
if test_corrupted:
    print("Corrupted test files:", test_corrupted)

print("\nTrain Audio Durations:")
print(train_audio_df['duration'].describe())

print("\nTest Audio Durations:")
print(test_audio_df['duration'].describe())

print("\nTrain Sample Rates:", train_audio_df['samplerate'].value_counts().to_dict())
print("Train Channels:", train_audio_df['channels'].value_counts().to_dict())
print("Train Formats:", train_audio_df['format'].value_counts().to_dict(), train_audio_df['subtype'].value_counts().to_dict())

print("\nTest Sample Rates:", test_audio_df['samplerate'].value_counts().to_dict())
print("Test Channels:", test_audio_df['channels'].value_counts().to_dict())

# Save inspection summary
train_audio_df.to_csv(os.path.join(output_dir, "train_audio_meta.csv"), index=False)
test_audio_df.to_csv(os.path.join(output_dir, "test_audio_meta.csv"), index=False)

# Visualizations
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
sns.histplot(train_df['label'], bins=11, kde=True, color='royalblue', discrete=False)
plt.title("Distribution of Grammar Scores (Train)")
plt.xlabel("Score (0.0 - 5.0)")
plt.ylabel("Count")
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
sns.histplot(train_audio_df['duration'], bins=30, kde=True, color='teal', label='Train')
sns.histplot(test_audio_df['duration'], bins=30, kde=True, color='coral', label='Test', alpha=0.6)
plt.title("Audio Duration Distribution (Seconds)")
plt.xlabel("Duration (s)")
plt.ylabel("Count")
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(output_dir, "eda_target_and_duration.png"), dpi=200)
plt.close()

print(f"\nInspection complete! Metadata and plots saved to {output_dir}")
