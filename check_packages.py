import sys

packages = [
    'numpy', 'scipy', 'pandas', 'sklearn', 'matplotlib', 'seaborn',
    'librosa', 'soundfile', 'audioread', 'torchaudio', 'torch',
    'transformers', 'whisper', 'faster_whisper', 'sentence_transformers',
    'nltk', 'spacy', 'language_tool_python', 'textstat',
    'lightgbm', 'xgboost', 'catboost', 'joblib', 'tqdm'
]

results = {}
for pkg in packages:
    try:
        mod = __import__(pkg)
        ver = getattr(mod, '__version__', 'installed')
        results[pkg] = ver
    except Exception as e:
        results[pkg] = f"MISSING ({type(e).__name__})"

print(f"Python: {sys.version}")
print("-" * 50)
for k, v in results.items():
    print(f"{k:25}: {v}")
