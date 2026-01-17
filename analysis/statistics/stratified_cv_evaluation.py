#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Stratified cross-validation evaluation using structured labels as proxy for external validation."""
import os
import json
import argparse
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, recall_score, precision_score, confusion_matrix
from sklearn.linear_model import LogisticRegression


def load_dataset(data_path: Path) -> pd.DataFrame:
    train_csv = data_path / 'train_labels.csv'
    test_csv = data_path / 'test_labels.csv'
    frames = []
    if train_csv.exists():
        frames.append(pd.read_csv(train_csv))
    if test_csv.exists():
        frames.append(pd.read_csv(test_csv))
    if not frames:
        raise FileNotFoundError('No train_labels.csv or test_labels.csv found at given data path')
    df = pd.concat(frames, ignore_index=True)
    # cleanup column names
    df.columns = [c.strip() for c in df.columns]
    df = df.dropna(subset=['label'])
    df['label'] = df['label'].astype(int)
    return df


def build_pipeline(categorical: List[str], numeric: List[str]) -> Pipeline:
    transformers = []
    if categorical:
        cat_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ])
        transformers.append(('cat', cat_transformer, categorical))
    if numeric:
        num_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])
        transformers.append(('num', num_transformer, numeric))
    if not transformers:
        raise ValueError('No features available for modeling')
    preprocessor = ColumnTransformer(transformers=transformers)
    clf = LogisticRegression(max_iter=200, class_weight='balanced')
    pipe = Pipeline(steps=[('preprocess', preprocessor), ('clf', clf)])
    return pipe


def compute_metrics(y_true, y_prob, threshold=0.5) -> Dict[str, float]:
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    sensitivity = tp / (tp + fn + 1e-9)
    specificity = tn / (tn + fp + 1e-9)
    ppv = tp / (tp + fp + 1e-9)
    npv = tn / (tn + fn + 1e-9)
    biopsy_rate = y_pred.mean()
    metrics = {
        'auc': roc_auc_score(y_true, y_prob),
        'accuracy': accuracy_score(y_true, y_pred),
        'f1': f1_score(y_true, y_pred),
        'sensitivity': sensitivity,
        'specificity': specificity,
        'ppv': ppv,
        'npv': npv,
        'biopsy_rate': biopsy_rate,
        'recall': recall_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred)
    }
    return metrics


def bootstrap_ci(values: List[float], n_bootstrap: int = 1000, ci: float = 0.95) -> Dict[str, float]:
    if len(values) == 0:
        return {'mean': None, 'lower': None, 'upper': None}
    arr = np.array(values)
    boot = []
    for _ in range(n_bootstrap):
        sample = np.random.choice(arr, size=len(arr), replace=True)
        boot.append(sample.mean())
    lower = np.percentile(boot, (1-ci) / 2 * 100)
    upper = np.percentile(boot, (1 + ci) / 2 * 100)
    return {'mean': float(arr.mean()), 'lower': float(lower), 'upper': float(upper)}


def main():
    parser = argparse.ArgumentParser(description='Stratified CV evaluation for external validation proxy')
    parser.add_argument('--data_path', type=str, default='5centers_multi', help='Path to data directory')
    parser.add_argument('--n_splits', type=int, default=5)
    parser.add_argument('--output', type=str, default='analysis/statistics/stratified_cv_results.json')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    data_path = Path(args.data_path)
    df = load_dataset(data_path)

    feature_candidates = [col for col in df.columns if col not in ['label', 'ID', 'OCT']]
    categorical = []
    numeric = []
    for col in feature_candidates:
        if df[col].dtype == object or df[col].dtype == 'O':
            categorical.append(col)
        else:
            numeric.append(col)
    pipe = build_pipeline(categorical, numeric)

    skf = StratifiedKFold(n_splits=args.n_splits, shuffle=True, random_state=args.seed)
    metrics_per_fold = []
    fold_results = []
    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(df, df['label'])):
        train_df = df.iloc[train_idx]
        test_df = df.iloc[test_idx]
        pipe.fit(train_df[feature_candidates], train_df['label'])
        y_prob = pipe.predict_proba(test_df[feature_candidates])[:, 1]
        metrics = compute_metrics(test_df['label'].values, y_prob)
        metrics_per_fold.append(metrics)
        metrics['fold'] = fold_idx + 1
        fold_results.append(metrics)
        print(f"Fold {fold_idx+1}: AUC={metrics['auc']:.3f}, Sens={metrics['sensitivity']:.3f}, Spec={metrics['specificity']:.3f}")

    aggregate = {}
    for key in metrics_per_fold[0].keys():
        values = [m[key] for m in metrics_per_fold]
        aggregate[key] = bootstrap_ci(values)

    output = {
        'n_splits': args.n_splits,
        'data_path': str(data_path),
        'features': {
            'categorical': categorical,
            'numeric': numeric
        },
        'fold_metrics': fold_results,
        'aggregate_metrics': aggregate
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"✅ Stratified CV results saved to: {output_path}")


if __name__ == '__main__':
    main()
