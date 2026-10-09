import os
import re
import math
import numpy as np
import pandas as pd
import nltk
from nltk import pos_tag, word_tokenize, sent_tokenize
import textstat
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD

FILLER_WORDS = {
    'uh', 'um', 'ah', 'er', 'eh', 'like', 'well', 'actually', 'basically',
    'literally', 'honestly', 'right', 'okay', 'ok', 'yeah'
}

vader_analyzer = SentimentIntensityAnalyzer()

def compute_single_text_features(text, duration=0.0):
    text = str(text) if pd.notnull(text) else ""
    text_clean = text.strip()
    
    # Defaults
    features = {}
    
    if not text_clean:
        # Return zeros for empty text
        base_keys = [
            'char_count', 'word_count', 'unique_word_count', 'ttr', 'root_ttr',
            'brunet_w', 'avg_word_length', 'std_word_length', 'long_words_ratio',
            'sentence_count', 'avg_sentence_length', 'std_sentence_length',
            'syllable_count', 'avg_syllables_per_word',
            'flesch_reading_ease', 'flesch_kincaid_grade', 'gunning_fog',
            'coleman_liau', 'automated_readability', 'dale_chall',
            'noun_ratio', 'verb_ratio', 'adj_ratio', 'adv_ratio', 'pronoun_ratio',
            'prep_ratio', 'conj_ratio', 'modal_ratio', 'past_verb_ratio', 'pres_verb_ratio',
            'filler_count', 'filler_ratio', 'repetition_count', 'repetition_ratio',
            'wpm_text', 'wps_text',
            'vader_pos', 'vader_neg', 'vader_neu', 'vader_compound'
        ]
        return {k: 0.0 for k in base_keys}

    # Tokenization
    words = [w.lower() for w in re.findall(r'\b[a-zA-Z]+\b', text_clean)]
    sentences = re.split(r'[.!?]+', text_clean)
    sentences = [s.strip() for s in sentences if len(s.strip()) > 0]
    
    N_words = len(words)
    N_chars = len(text_clean)
    N_unique = len(set(words))
    N_sentences = max(1, len(sentences))
    
    # 1. Lexical Diversity
    ttr = (N_unique / N_words) if N_words > 0 else 0.0
    root_ttr = (N_unique / math.sqrt(N_words)) if N_words > 0 else 0.0
    brunet_w = (N_words ** (N_unique ** -0.172)) if (N_words > 0 and N_unique > 0) else 0.0
    
    word_lens = [len(w) for w in words] if words else [0]
    avg_word_len = float(np.mean(word_lens))
    std_word_len = float(np.std(word_lens))
    long_words = sum(1 for w in words if len(w) > 6)
    long_words_ratio = (long_words / N_words) if N_words > 0 else 0.0
    
    # 2. Sentences
    sent_lens = [len(re.findall(r'\b[a-zA-Z]+\b', s)) for s in sentences] if sentences else [0]
    avg_sent_len = float(np.mean(sent_lens))
    std_sent_len = float(np.std(sent_lens))
    
    # 3. Readability & Syllables
    try:
        syllables = textstat.syllable_count(text_clean)
        avg_syllables = syllables / max(1, N_words)
        f_ease = textstat.flesch_reading_ease(text_clean)
        f_grade = textstat.flesch_kincaid_grade(text_clean)
        g_fog = textstat.gunning_fog(text_clean)
        c_liau = textstat.coleman_liau_index(text_clean)
        ari = textstat.automated_readability_index(text_clean)
        d_chall = textstat.dale_chall_readability_score(text_clean)
    except Exception:
        syllables, avg_syllables = 0, 0.0
        f_ease, f_grade, g_fog, c_liau, ari, d_chall = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        
    # 4. POS tags
    pos_tags = pos_tag(words) if words else []
    tag_counts = {
        'noun': 0, 'verb': 0, 'adj': 0, 'adv': 0, 'pronoun': 0,
        'prep': 0, 'conj': 0, 'modal': 0, 'past_verb': 0, 'pres_verb': 0
    }
    for w, t in pos_tags:
        if t.startswith('NN'): tag_counts['noun'] += 1
        elif t.startswith('VB'):
            tag_counts['verb'] += 1
            if t in ('VBD', 'VBN'): tag_counts['past_verb'] += 1
            elif t in ('VBP', 'VBZ', 'VBG'): tag_counts['pres_verb'] += 1
        elif t.startswith('JJ'): tag_counts['adj'] += 1
        elif t.startswith('RB'): tag_counts['adv'] += 1
        elif t.startswith('PRP') or t.startswith('WP'): tag_counts['pronoun'] += 1
        elif t == 'IN': tag_counts['prep'] += 1
        elif t == 'CC': tag_counts['conj'] += 1
        elif t == 'MD': tag_counts['modal'] += 1
        
    noun_ratio = tag_counts['noun'] / max(1, N_words)
    verb_ratio = tag_counts['verb'] / max(1, N_words)
    adj_ratio = tag_counts['adj'] / max(1, N_words)
    adv_ratio = tag_counts['adv'] / max(1, N_words)
    pronoun_ratio = tag_counts['pronoun'] / max(1, N_words)
    prep_ratio = tag_counts['prep'] / max(1, N_words)
    conj_ratio = tag_counts['conj'] / max(1, N_words)
    modal_ratio = tag_counts['modal'] / max(1, N_words)
    past_verb_ratio = tag_counts['past_verb'] / max(1, N_words)
    pres_verb_ratio = tag_counts['pres_verb'] / max(1, N_words)
    
    # 5. Fluency & Repetitions
    filler_count = sum(1 for w in words if w in FILLER_WORDS)
    filler_ratio = filler_count / max(1, N_words)
    
    repetition_count = sum(1 for i in range(len(words)-1) if words[i] == words[i+1])
    repetition_ratio = repetition_count / max(1, N_words)
    
    # Speech rate
    wpm_text = (N_words / (duration / 60.0)) if duration > 0 else 0.0
    wps_text = (N_words / duration) if duration > 0 else 0.0
    
    # 6. Sentiment
    vader = vader_analyzer.polarity_scores(text_clean)
    
    return {
        'char_count': float(N_chars),
        'word_count': float(N_words),
        'unique_word_count': float(N_unique),
        'ttr': float(ttr),
        'root_ttr': float(root_ttr),
        'brunet_w': float(brunet_w),
        'avg_word_length': float(avg_word_len),
        'std_word_length': float(std_word_len),
        'long_words_ratio': float(long_words_ratio),
        'sentence_count': float(N_sentences),
        'avg_sentence_length': float(avg_sent_len),
        'std_sentence_length': float(std_sent_len),
        'syllable_count': float(syllables),
        'avg_syllables_per_word': float(avg_syllables),
        'flesch_reading_ease': float(f_ease),
        'flesch_kincaid_grade': float(f_grade),
        'gunning_fog': float(g_fog),
        'coleman_liau': float(c_liau),
        'automated_readability': float(ari),
        'dale_chall': float(d_chall),
        'noun_ratio': float(noun_ratio),
        'verb_ratio': float(verb_ratio),
        'adj_ratio': float(adj_ratio),
        'adv_ratio': float(adv_ratio),
        'pronoun_ratio': float(pronoun_ratio),
        'prep_ratio': float(prep_ratio),
        'conj_ratio': float(conj_ratio),
        'modal_ratio': float(modal_ratio),
        'past_verb_ratio': float(past_verb_ratio),
        'pres_verb_ratio': float(pres_verb_ratio),
        'filler_count': float(filler_count),
        'filler_ratio': float(filler_ratio),
        'repetition_count': float(repetition_count),
        'repetition_ratio': float(repetition_ratio),
        'wpm_text': float(wpm_text),
        'wps_text': float(wps_text),
        'vader_pos': float(vader['pos']),
        'vader_neg': float(vader['neg']),
        'vader_neu': float(vader['neu']),
        'vader_compound': float(vader['compound']),
    }

def extract_all_text_features(train_trans_df, test_trans_df):
    train_records = []
    for _, row in train_trans_df.iterrows():
        dur = row.get('duration', 0.0)
        feat = compute_single_text_features(row.get('transcript', ''), dur)
        feat['filename'] = row['filename']
        feat['label'] = row.get('label', np.nan)
        train_records.append(feat)
        
    test_records = []
    for _, row in test_trans_df.iterrows():
        dur = row.get('duration', 0.0)
        feat = compute_single_text_features(row.get('transcript', ''), dur)
        feat['filename'] = row['filename']
        feat['label'] = row.get('label', np.nan)
        test_records.append(feat)
        
    df_train_txt = pd.DataFrame(train_records)
    df_test_txt = pd.DataFrame(test_records)
    
    # Add TF-IDF with SVD decomposition (10 dense semantic topics)
    print("Fitting TF-IDF + TruncatedSVD on transcripts...")
    all_transcripts = train_trans_df['transcript'].fillna('').tolist() + test_trans_df['transcript'].fillna('').tolist()
    tfidf = TfidfVectorizer(max_features=500, stop_words='english', min_df=2, ngram_range=(1,2))
    tfidf_matrix = tfidf.fit_transform(all_transcripts)
    
    n_components = min(15, tfidf_matrix.shape[1] - 1)
    svd = TruncatedSVD(n_components=n_components, random_state=42)
    svd_feats = svd.fit_transform(tfidf_matrix)
    
    train_len = len(train_trans_df)
    train_svd = svd_feats[:train_len]
    test_svd = svd_feats[train_len:]
    
    for k in range(n_components):
        df_train_txt[f'tfidf_svd_{k}'] = train_svd[:, k]
        df_test_txt[f'tfidf_svd_{k}'] = test_svd[:, k]
        
    return df_train_txt, df_test_txt

if __name__ == '__main__':
    features_dir = r"c:\Users\chand\Downloads\shl-hiring-assessment-2026\features"
    train_trans_path = os.path.join(features_dir, "transcripts_train.csv")
    test_trans_path = os.path.join(features_dir, "transcripts_test.csv")
    
    if os.path.exists(train_trans_path) and os.path.exists(test_trans_path):
        train_trans_df = pd.read_csv(train_trans_path)
        test_trans_df = pd.read_csv(test_trans_path)
        
        df_train_txt, df_test_txt = extract_all_text_features(train_trans_df, test_trans_df)
        df_train_txt.to_csv(os.path.join(features_dir, "text_features_train.csv"), index=False)
        df_test_txt.to_csv(os.path.join(features_dir, "text_features_test.csv"), index=False)
        print(f"Saved text features: Train {df_train_txt.shape}, Test {df_test_txt.shape}")
    else:
        print("Transcripts not found yet. Run extract_transcripts.py first.")
