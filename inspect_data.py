import os
import pandas as pd
import numpy as np

base_dir = r"c:\Users\chand\Downloads\shl-hiring-assessment-2026\Dataset_Final"
train_csv_path = os.path.join(base_dir, "train.csv")
test_csv_path = os.path.join(base_dir, "test.csv")
sample_sub_path = os.path.join(base_dir, "sample_submission.csv")
train_audio_dir = os.path.join(base_dir, "train")
test_audio_dir = os.path.join(base_dir, "test")

train_df = pd.read_csv(train_csv_path)
test_df = pd.read_csv(test_csv_path)
sample_sub_df = pd.read_csv(sample_sub_path)

print(f"train_df shape: {train_df.shape}")
print(f"test_df shape: {test_df.shape}")
print(f"sample_sub_df shape: {sample_sub_df.shape}")

print("\n--- train.csv info ---")
print(train_df.info())
print("\n--- train label statistics ---")
print(train_df['label'].describe())
print("Unique labels in train:", sorted(train_df['label'].unique()))
print("Missing values in train:", train_df.isnull().sum().to_dict())
print("Duplicate filenames in train:", train_df['filename'].duplicated().sum())

print("\n--- test.csv info ---")
print(test_df.info())
print("Missing values in test:", test_df.isnull().sum().to_dict())
print("Duplicate filenames in test:", test_df['filename'].duplicated().sum())

print("\n--- sample_submission.csv info ---")
print(sample_sub_df.info())
print("Missing values in sample_sub:", sample_sub_df.isnull().sum().to_dict())

# Compare filenames between test_df and sample_sub_df
test_files = set(test_df['filename'])
sub_files = set(sample_sub_df['filename'])
print(f"\nOverlap between test.csv and sample_submission.csv: {len(test_files.intersection(sub_files))} files")
print(f"In test.csv but not sample_sub: {len(test_files - sub_files)}")
print(f"In sample_sub but not test.csv: {len(sub_files - test_files)}")

# Check audio directories
train_audio_files = os.listdir(train_audio_dir) if os.path.exists(train_audio_dir) else []
test_audio_files = os.listdir(test_audio_dir) if os.path.exists(test_audio_dir) else []
print(f"\nFiles in train audio dir: {len(train_audio_files)}")
print(f"Files in test audio dir: {len(test_audio_files)}")

# Check audio file existence
missing_train_audio = [f for f in train_df['filename'] if not os.path.exists(os.path.join(train_audio_dir, f))]
missing_test_audio = [f for f in test_df['filename'] if not os.path.exists(os.path.join(test_audio_dir, f))]
missing_sub_audio = [f for f in sample_sub_df['filename'] if not os.path.exists(os.path.join(test_audio_dir, f))]

print(f"Train CSV filenames missing audio files: {len(missing_train_audio)}")
print(f"Test CSV filenames missing audio files: {len(missing_test_audio)}")
print(f"Sample sub filenames missing audio in test dir: {len(missing_sub_audio)}")

# Also check if sample_sub filenames exist in train audio dir
sub_in_train = [f for f in sample_sub_df['filename'] if os.path.exists(os.path.join(train_audio_dir, f))]
print(f"Sample sub filenames found in train audio dir: {len(sub_in_train)}")
