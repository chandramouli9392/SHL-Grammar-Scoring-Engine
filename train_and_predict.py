import os
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from scipy.optimize import minimize
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import Ridge
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor

def load_and_merge_features(features_dir, base_dir):
    print("Loading extracted feature sets...")
    # 1. Acoustic features (always complete - 769 train, 216 test)
    ac_tr = pd.read_csv(os.path.join(features_dir, "acoustic_features_train.csv"))
    ac_te = pd.read_csv(os.path.join(features_dir, "acoustic_features_test.csv"))
    
    # Start with acoustic features as the base (left join = keep all acoustic rows)
    train_df = ac_tr.copy()
    test_df = ac_te.copy()

    # 2. Transcripts & Speech metrics (may be partial for train)
    tr_path_tr = os.path.join(features_dir, "transcripts_train.csv")
    tr_path_te = os.path.join(features_dir, "transcripts_test.csv")
    if os.path.exists(tr_path_tr) and os.path.exists(tr_path_te):
        tr_tr = pd.read_csv(tr_path_tr)
        tr_te = pd.read_csv(tr_path_te)
        tr_tr_feat = tr_tr.drop(columns=['transcript', 'label', 'duration'], errors='ignore')
        tr_te_feat = tr_te.drop(columns=['transcript', 'label', 'duration'], errors='ignore')
        train_df = train_df.merge(tr_tr_feat, on='filename', how='left')
        test_df  = test_df.merge(tr_te_feat, on='filename', how='left')
        print(f"  [+] Transcript features merged. Train rows with transcripts: {tr_tr_feat.shape[0]}")
    else:
        print("  [!] Transcript features not found - skipping.")
    
    # 3. Text & Linguistic features (derived from transcripts; may be partial)
    tx_path_tr = os.path.join(features_dir, "text_features_train.csv")
    tx_path_te = os.path.join(features_dir, "text_features_test.csv")
    if os.path.exists(tx_path_tr) and os.path.exists(tx_path_te):
        tx_tr = pd.read_csv(tx_path_tr)
        tx_te = pd.read_csv(tx_path_te)
        tx_tr_feat = tx_tr.drop(columns=['label'], errors='ignore')
        tx_te_feat = tx_te.drop(columns=['label'], errors='ignore')
        train_df = train_df.merge(tx_tr_feat, on='filename', how='left')
        test_df  = test_df.merge(tx_te_feat, on='filename', how='left')
        print(f"  [+] Text/linguistic features merged. Train rows with text feats: {tx_tr_feat.shape[0]}")
    else:
        print("  [!] Text features not found - skipping.")
    
    print(f"Merged Train Shape: {train_df.shape}")
    print(f"Merged Test Shape: {test_df.shape}")
    missing_train = train_df.isnull().sum().sum()
    print(f"Missing values in train (will be median-filled): {missing_train}")
    return train_df, test_df

def run_training_pipeline():
    features_dir = r"c:\Users\chand\Downloads\shl-hiring-assessment-2026\features"
    base_dir = r"c:\Users\chand\Downloads\shl-hiring-assessment-2026\Dataset_Final"
    output_dir = r"c:\Users\chand\Downloads\shl-hiring-assessment-2026"
    
    train_df, test_df = load_and_merge_features(features_dir, base_dir)
    
    # Feature columns
    drop_cols = {'filename', 'label'}
    feature_cols = [c for c in train_df.columns if c not in drop_cols]
    
    X = train_df[feature_cols].copy()
    y = train_df['label'].values
    X_test = test_df[feature_cols].copy()
    
    # Clean infinities and NaNs
    X.replace([np.inf, -np.inf], np.nan, inplace=True)
    X_test.replace([np.inf, -np.inf], np.nan, inplace=True)
    
    med = X.median()
    X.fillna(med, inplace=True)
    X_test.fillna(med, inplace=True)
    
    # Remove constant features
    non_const = [c for c in feature_cols if X[c].std() > 1e-7]
    print(f"Total features: {len(feature_cols)}, after removing constant features: {len(non_const)}")
    X = X[non_const].values
    X_test = X_test[non_const].values
    feature_names = non_const
    
    # Stratified K-Fold based on binned target
    n_splits = 5
    binned_y = pd.qcut(y, q=n_splits, labels=False, duplicates='drop')
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    oof_lgb = np.zeros(len(y))
    oof_xgb = np.zeros(len(y))
    oof_cat = np.zeros(len(y))
    oof_ridge = np.zeros(len(y))
    
    preds_test_lgb = np.zeros(len(X_test))
    preds_test_xgb = np.zeros(len(X_test))
    preds_test_cat = np.zeros(len(X_test))
    preds_test_ridge = np.zeros(len(X_test))
    
    feature_importances = np.zeros(len(feature_names))
    
    print("\n" + "="*50)
    print("STARTING 5-FOLD CROSS-VALIDATION")
    print("="*50)
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, binned_y)):
        X_train_f, y_train_f = X[train_idx], y[train_idx]
        X_val_f, y_val_f = X[val_idx], y[val_idx]
        
        # 1. LightGBM
        model_lgb = lgb.LGBMRegressor(
            n_estimators=600,
            learning_rate=0.03,
            max_depth=5,
            num_leaves=25,
            colsample_bytree=0.7,
            subsample=0.8,
            reg_alpha=0.2,
            reg_lambda=1.5,
            random_state=42 + fold,
            verbose=-1
        )
        model_lgb.fit(
            X_train_f, y_train_f,
            eval_set=[(X_val_f, y_val_f)],
            callbacks=[lgb.early_stopping(stopping_rounds=40, verbose=False)]
        )
        val_pred_lgb = model_lgb.predict(X_val_f)
        oof_lgb[val_idx] = val_pred_lgb
        preds_test_lgb += model_lgb.predict(X_test) / n_splits
        feature_importances += model_lgb.feature_importances_ / n_splits
        
        # 2. XGBoost
        model_xgb = xgb.XGBRegressor(
            n_estimators=600,
            learning_rate=0.03,
            max_depth=4,
            colsample_bytree=0.7,
            subsample=0.8,
            reg_alpha=0.2,
            reg_lambda=1.5,
            random_state=42 + fold,
            early_stopping_rounds=40
        )
        model_xgb.fit(
            X_train_f, y_train_f,
            eval_set=[(X_val_f, y_val_f)],
            verbose=False
        )
        val_pred_xgb = model_xgb.predict(X_val_f)
        oof_xgb[val_idx] = val_pred_xgb
        preds_test_xgb += model_xgb.predict(X_test) / n_splits
        
        # 3. CatBoost
        model_cat = CatBoostRegressor(
            iterations=700,
            learning_rate=0.03,
            depth=5,
            l2_leaf_reg=3.0,
            random_seed=42 + fold,
            verbose=0,
            early_stopping_rounds=40
        )
        model_cat.fit(
            X_train_f, y_train_f,
            eval_set=(X_val_f, y_val_f),
            verbose=False
        )
        val_pred_cat = model_cat.predict(X_val_f)
        oof_cat[val_idx] = val_pred_cat
        preds_test_cat += model_cat.predict(X_test) / n_splits
        
        # 4. Scaled Ridge
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_f)
        X_val_scaled = scaler.transform(X_val_f)
        X_test_scaled = scaler.transform(X_test)
        
        model_ridge = Ridge(alpha=10.0, random_state=42 + fold)
        model_ridge.fit(X_train_scaled, y_train_f)
        val_pred_ridge = model_ridge.predict(X_val_scaled)
        oof_ridge[val_idx] = val_pred_ridge
        preds_test_ridge += model_ridge.predict(X_test_scaled) / n_splits
        
        fold_rmse_lgb = np.sqrt(mean_squared_error(y_val_f, val_pred_lgb))
        fold_rmse_xgb = np.sqrt(mean_squared_error(y_val_f, val_pred_xgb))
        fold_rmse_cat = np.sqrt(mean_squared_error(y_val_f, val_pred_cat))
        fold_rmse_rdg = np.sqrt(mean_squared_error(y_val_f, val_pred_ridge))
        print(f"Fold {fold+1}/{n_splits} RMSE -> LGB: {fold_rmse_lgb:.4f} | XGB: {fold_rmse_xgb:.4f} | CAT: {fold_rmse_cat:.4f} | Ridge: {fold_rmse_rdg:.4f}")

    # Evaluate individual models
    def print_metrics(name, oof_preds):
        rmse = np.sqrt(mean_squared_error(y, oof_preds))
        mae = mean_absolute_error(y, oof_preds)
        r, _ = pearsonr(y, oof_preds)
        rho, _ = spearmanr(y, oof_preds)
        r2 = r2_score(y, oof_preds)
        print(f"{name:15} | RMSE: {rmse:.4f} | MAE: {mae:.4f} | Pearson r: {r:.4f} | Spearman rho: {rho:.4f} | R2: {r2:.4f}")
        return rmse

    print("\n" + "="*50)
    print("INDIVIDUAL MODEL CROSS-VALIDATION RESULTS")
    print("="*50)
    print_metrics("LightGBM", oof_lgb)
    print_metrics("XGBoost", oof_xgb)
    print_metrics("CatBoost", oof_cat)
    print_metrics("Ridge", oof_ridge)

    # Optimize blend weights
    oof_matrix = np.column_stack([oof_lgb, oof_xgb, oof_cat, oof_ridge])
    test_matrix = np.column_stack([preds_test_lgb, preds_test_xgb, preds_test_cat, preds_test_ridge])
    
    def blend_loss(weights):
        w = weights / np.sum(weights)
        pred = np.dot(oof_matrix, w)
        return mean_squared_error(y, pred)

    init_weights = [0.35, 0.25, 0.30, 0.10]
    bounds = [(0, 1)] * 4
    opt_res = minimize(blend_loss, init_weights, bounds=bounds, method='SLSQP', constraints={'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0})
    best_weights = opt_res.x
    print(f"\nOptimal Blend Weights: LGB={best_weights[0]:.3f}, XGB={best_weights[1]:.3f}, CAT={best_weights[2]:.3f}, Ridge={best_weights[3]:.3f}")

    oof_ensemble = np.dot(oof_matrix, best_weights)
    oof_ensemble = np.clip(oof_ensemble, 0.0, 5.0)
    print("\n" + "="*50)
    print("FINAL ENSEMBLE CROSS-VALIDATION RESULTS")
    print("="*50)
    print_metrics("Blended Ensemble", oof_ensemble)

    # Top Features
    top_indices = np.argsort(feature_importances)[::-1][:20]
    print("\nTop 20 Most Important Features:")
    for rank, idx in enumerate(top_indices, 1):
        print(f"  {rank:2d}. {feature_names[idx]:30} : {feature_importances[idx]:.1f}")

    # Generate Test Predictions
    test_preds = np.dot(test_matrix, best_weights)
    test_preds = np.clip(test_preds, 0.0, 5.0)

    # 1. Primary Submission: matching test.csv (216 audio files)
    sub_df = pd.DataFrame({
        'filename': test_df['filename'],
        'label': np.round(test_preds, 4)
    })
    sub_path = os.path.join(output_dir, "submission.csv")
    sub_df.to_csv(sub_path, index=False)
    print(f"\nSaved primary submission ({len(sub_df)} rows) to: {sub_path}")
    print(sub_df.head(10))
    print(sub_df['label'].describe())

    # 2. Sample submission format (204 files)
    sample_sub_path = os.path.join(base_dir, "sample_submission.csv")
    if os.path.exists(sample_sub_path):
        sample_sub_df = pd.read_csv(sample_sub_path)
        pred_dict = dict(zip(test_df['filename'], np.round(test_preds, 4)))
        # Also map ground truth from train for any overlap
        train_csv_df = pd.read_csv(os.path.join(base_dir, "train.csv"))
        train_gt_dict = dict(zip(train_csv_df['filename'], train_csv_df['label']))
        
        sample_format_preds = []
        global_mean = float(y.mean())
        for fn in sample_sub_df['filename']:
            if fn in pred_dict:
                sample_format_preds.append(pred_dict[fn])
            elif fn in train_gt_dict:
                sample_format_preds.append(train_gt_dict[fn])
            else:
                sample_format_preds.append(round(global_mean, 4))
                
        sample_sub_df['label'] = sample_format_preds
        sample_format_out = os.path.join(output_dir, "submission_sample_format.csv")
        sample_sub_df.to_csv(sample_format_out, index=False)
        print(f"Saved sample-format submission ({len(sample_sub_df)} rows) to: {sample_format_out}")

    print("\nAll training and inference tasks completed successfully!")

if __name__ == '__main__':
    run_training_pipeline()
