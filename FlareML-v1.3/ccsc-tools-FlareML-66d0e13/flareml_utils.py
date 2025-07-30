'''
 (c) Copyright 2021
 All rights reserved
 Programs written by Yasser Abduallah
 Department of Computer Science
 New Jersey Institute of Technology
 University Heights, Newark, NJ 07102, USA

 Permission to use, copy, modify, and distribute this
 software and its documentation for any purpose and without
 fee is hereby granted, provided that this copyright
 notice appears in all copies. Programmer(s) makes no
 representations about the suitability of this
 software for any purpose.  It is provided "as is" without
 express or implied warranty.

 @author: Yasser Abduallah
 
 ---------------------------------------------------------------------
 This code has been modified to work with FlareML v1.3. 
 The modified version is called FlareBlueML. 
 Modifications by Daniel Graves, July 2025.
 ---------------------------------------------------------------------
'''

from __future__ import division
from xgboost import XGBClassifier
import warnings
warnings.filterwarnings('ignore')
import math
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
import time
import pickle
from contextlib import contextmanager
from random import uniform
import os
import datetime
from pathlib import Path
from os import listdir
from os.path import isfile, join

# Scikit-learn imports
from sklearn.model_selection import (
    train_test_split,
    cross_val_score,
    KFold,
    GridSearchCV,
    TimeSeriesSplit
)
from sklearn.metrics import accuracy_score, confusion_matrix, multilabel_confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
# from sklearn.neural_network import MLPClassifier  # MLP REMOVED
from sklearn.linear_model import LogisticRegression

# Third-party model imports
from sklearn_extensions.extreme_learning_machines.elm import GenELMClassifier
from sklearn_extensions.extreme_learning_machines.random_layer import RBFRandomLayer, MLPRandomLayer
from lightgbm import LGBMClassifier

# Directory settings
custom_models_dir = "custom_models"
custom_models_data_dir = "custom_models_data"
custom_models_time_limit = 24 * 60  # 24 hours in minutes
default_models_dir = "models"

# Supported algorithms
algorithms = ['ENS', 'RF', 'ELM', 'LGBM', 'XGB']  # MLP REMOVED
algorithms_names = [
    'Ensemble',
    'Random Forest',
    # 'Multiple Layer Perceptron (MLP)',  # MLP REMOVED
    'Extreme Learning Machine (ELM)',
    'Light Gradient Boosting (LGBM)',
    'Extreme Gradient Boosting (XGB)'
]

# Global variables
timestr = time.strftime("%Y%m%d_%H%M%S")
loggingString = []
overall_test_accuracy = None
partial_ens_trained = False
noLogging = False
log_to_terminal = False
verbose = False
save_stdout = sys.stdout

flares_col_name = 'Flare Class'
logFile = "logs/ens_deepsun.log"
mapping = {1: "B", 2: "C", 3: "M", 4: "X", -1: "N/A"}
class_to_num = {v: k for k, v in mapping.items()}
req_columns = [
    flares_col_name, "TOTUSJH", "TOTBSQ", "TOTPOT", "TOTUSJZ",
    "ABSNJZH", "SAVNCPP", "USFLUX", "AREA_ACR", "TOTFZ",
    "MEANPOT", "R_VALUE", "EPSZ", "SHRGT45"
]

# Context managers for stdout redirection
@contextmanager
def stdout_redirected(new_stdout):
    global save_stdout
    save_stdout = sys.stdout
    sys.stdout = new_stdout
    try:
        yield
    finally:
        sys.stdout = save_stdout

@contextmanager
def stdout_default():
    global save_stdout
    sys.stdout = save_stdout
    yield

# Logging utility
def log(*message, verbose=True, logToTerminal=False, no_time=False, end=' '):
    global noLogging, log_to_terminal, loggingString
    if noLogging:
        return
    if log_to_terminal or logToTerminal:
        if not no_time:
            print(f'[{datetime.datetime.now().replace(microsecond=0)}]', end=end)
        print(*message, end=end)
        print()
    with open(logFile, "a+") as lf:
        with stdout_redirected(lf):
            if not no_time:
                print(f'[{datetime.datetime.now().replace(microsecond=0)}]', end=end)
            for msg in message:
                print(msg, end=end)
                if verbose:
                    loggingString.append(str(msg))
            print()

def set_log_to_terminal(v):
    global log_to_terminal
    log_to_terminal = bool(v)

def set_verbose(v):
    global verbose
    verbose = bool(v)

def boolean(b):
    if b is None:
        return False
    return str(b).strip().lower() in ['y', 'yes', '1', 'true']

# Model persistence helpers
def create_default_model(trained_model, model_id):
    return create_model(trained_model, model_id, default_models_dir)

def create_custom_model(trained_model, model_id):
    return create_model(trained_model, model_id, custom_models_dir)

def create_model(trained_model, model_id, model_dir):
    os.makedirs(model_dir, exist_ok=True)
    path = os.path.join(model_dir, f"{model_id}.sav")
    log(f"Saving model {model_id} to {path}")
    pickle.dump(trained_model, open(path, 'wb'))
    return path

# File existence checks
def is_file_exists(file):
    exists = Path(file).exists()
    log(f"Check if file exists: {file} : {exists}")
    return exists

def are_model_files_exist(models_dir, modelId, alg='ENS'):
    rf_exists = os.path.isfile(f"{models_dir}/{modelId}_rf.sav")
    elm_exists = os.path.isfile(f"{models_dir}/{modelId}_elm.sav")
    lgbm_exists = os.path.isfile(f"{models_dir}/{modelId}_lgbm.sav")
    xgb_exists = os.path.isfile(f"{models_dir}/{modelId}_xgb.sav")
    if alg == 'ENS':
        global partial_ens_trained
        # Flag if at least one model is trained
        partial_ens_trained = any([rf_exists, elm_exists, lgbm_exists, xgb_exists])
        # Full ensemble ready only if all four exist
        return rf_exists and elm_exists and lgbm_exists and xgb_exists
    else:
        return {
            'RF': rf_exists,
            'ELM': elm_exists,
            'LGBM': lgbm_exists,
            'XGB': xgb_exists
        }.get(alg, False)

def get_partial_ens_trained():
    global partial_ens_trained
    return partial_ens_trained

# Data utilities
def convert_class_to_num(c):
    return class_to_num.get(str(c)[0].upper(), -1)

def load_model(model_dir, model_id):
    path = os.path.join(model_dir, f"{model_id}.sav")
    log(f"Loading model file: {path}")
    if is_file_exists(path):
        return pickle.load(open(path, 'rb'))
    log("Model file not found.")
    return None

def load_dataset_csv(data_file):
    log(f"Reading data set from file: {data_file}")
    return pd.read_csv(data_file)

def removeDataColumn(col, data):
    if col in data.columns:
        return data.drop(col, axis=1)
    return data

def split_data(dataset, target_column='flarecn', test_percent=0.1):
    labels = np.array(dataset[target_column])
    data = removeDataColumn(target_column, dataset)
    cols = data.columns
    return train_test_split(data[cols], labels, test_size=test_percent)

def normalize_scale_data(d):
    arr = np.array(d)
    return (arr - arr.min()) / (arr.max() - arr.min())

# Advanced Feature Engineering
"""
Adding anomaly detection with temporal derivative features. 
    -Rolling Statistics such as mean, std, skew. 
    -AS (Anomaly Score): deviation from Rolling Mean
    -Velocity: first derivative
    -Acceleration: second derivative
"""
def engineer_temporal_features(df, window_size=6, features=None):
    """
    Adds rolling statistics, anomaly score, velocity, and acceleration features.
    """
    if 'fdate' not in df.columns:
        return df  # If no timestamp column, skip
    df['fdate'] = pd.to_datetime(df['fdate'])
    df = df.sort_values(by='fdate').reset_index(drop=True)
    if features is None:
        features = ['TOTUSJH', 'TOTBSQ', 'SAVNCPP', 'R_VALUE']
    for feature in features:
        # Rolling statistics
        roll = df[feature].rolling(window=window_size)
        df[f'{feature}_roll_mean'] = roll.mean()
        df[f'{feature}_roll_std'] = roll.std()
        df[f'{feature}_roll_skew'] = roll.skew()
        # Anomaly Score
        df[f'{feature}_anomaly'] = df[feature] - df[f'{feature}_roll_mean']
        # First and Second derivatives
        df[f'{feature}_velocity'] = df[feature].diff()
        df[f'{feature}_acceleration'] = df[feature].diff().diff()
    df = df.bfill().ffill()
    return df

# Training Wrappers (with TimeSeriesSplit)
def rf_train_model(train_x=None, test_x=None, train_y=None, test_y=None, model_id="default_model"):
    tscv = TimeSeriesSplit(n_splits=5)
    param_grid = {
        'n_estimators': [200, 500],
        'max_features': ['sqrt', 6],
        'max_depth': [10, 20, None],
        'class_weight': ['balanced']
    }
    gs = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=tscv, verbose=2, n_jobs=-1)
    print("Tuning Random Forest...")
    gs.fit(train_x, train_y)
    print(f"Best RF params: {gs.best_params_}")
    log(f"RF best CV score: {gs.best_score_:.3f}", logToTerminal=True)
    return model_train_wrapper('RF', gs, train_x, test_x, train_y, test_y, model_id)

# To train LGBM you must type flareml_train.py -a LGBM
def lgbm_train_model(train_x=None, test_x=None, train_y=None, test_y=None, model_id="default_model"):
    tscv = TimeSeriesSplit(n_splits=5)
    param_grid = {
        'n_estimators': [100, 200],
        'learning_rate': [0.05, 0.1],
        'num_leaves': [20, 31],
        'class_weight': ['balanced'],
        'min_child_samples': [5, 10, 20]
    }
    gs = GridSearchCV(LGBMClassifier(random_state=42, verbosity=-1), param_grid, cv=tscv, verbose=2, n_jobs=-1)
    print("Tuning LightGBM...")
    gs.fit(train_x, train_y)
    print(f"Best LGBM params: {gs.best_params_}")
    log(f"LGBM best CV score: {gs.best_score_:.3f}", logToTerminal=True)
    return model_train_wrapper('LGBM', gs, train_x, test_x, train_y, test_y, model_id)

def xgb_train_model(train_x=None, test_x=None, train_y=None, test_y=None, model_id="default_model"):
    tscv = TimeSeriesSplit(n_splits=5)
    param_grid = {
        'n_estimators': [100, 200],
        'learning_rate': [0.05, 0.1],
        'max_depth': [3, 6],
        'subsample': [0.8, 1.0],
        'colsample_bytree': [0.8, 1.0]
    }
    # Convert labels to 0-based for XGBoost
    train_y = train_y - 1
    if test_y is not None:
        test_y = test_y - 1
    gs = GridSearchCV(XGBClassifier(objective='multi:softmax', num_class=4, eval_metric='mlogloss', use_label_encoder=False, random_state=42),
                      param_grid, cv=tscv, verbose=2, n_jobs=-1)
    print("Tuning XGBoost...")
    gs.fit(train_x, train_y)
    print(f"Best XGBoost params: {gs.best_params_}")
    return model_train_wrapper('XGB', gs, train_x, test_x, train_y, test_y, model_id)

# MLP REMOVED

def elm_train_model(train_x=None, test_x=None, train_y=None, test_y=None, model_id="default_model"):
    # 1) build and fit the scaler on **training** data only
    scaler = StandardScaler().fit(train_x)

    # 2) transform both train and test sets
    train_x_scaled = scaler.transform(train_x)
    test_x_scaled  = scaler.transform(test_x) if test_x is not None else None

    # 3) train & save the ELM model
    trained = model_train_wrapper(
        'ELM', GenELMClassifier(hidden_layer=MLPRandomLayer(n_hidden=200, activation_func='tanh')),
        train_x_scaled, test_x_scaled, train_y, test_y, model_id
    )

    # 4) SAVE the scaler right next to the model file
    model_dir   = custom_models_dir if model_id != 'default_model' else default_models_dir
    scaler_path = os.path.join(model_dir, f"{model_id}_elm_scaler.sav")
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)

    return trained


# Core Training Wrapper
def model_train_wrapper(model_name, alg_model, train_x=None, test_x=None, train_y=None, test_y=None, model_id='default_model'):
    if hasattr(alg_model, 'best_estimator_'):
        trained_model = alg_model.best_estimator_
    else:
        trained_model = alg_model.fit(train_x, train_y)
    model_dir = custom_models_dir if model_id != 'default_model' else default_models_dir
    create_model(trained_model, f"{model_id}_{model_name.lower()}", model_dir)
    return trained_model

# Prediction and Metrics
def model_prediction_wrapper(
    model_name,
    alg_model=None,
    test_x=None,
    test_y=None,
    model_id='default_model',
    return_proba=False
):
    model_dir = custom_models_dir if model_id != 'default_model' else default_models_dir
    # Load the model (or use the provided one)
    model = alg_model if alg_model is not None else load_model(
        model_dir, f"{model_id}_{model_name.lower()}"
    )
    if model is None:
        raise FileNotFoundError(f"Model for {model_name} not found")

    # If this is the ELM model, load and apply its saved scaler
    if model_name.upper() == 'ELM':
        scaler_path = os.path.join(model_dir, f"{model_id}_elm_scaler.sav")
        if not os.path.isfile(scaler_path):
            raise FileNotFoundError(
                "ELM scaler not found; please retrain the ELM model first."
            )
        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)
        test_x = scaler.transform(test_x)

    # If user wants probabilities, only return if the model supports it
    if return_proba:
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(test_x)
            # ensure it's a 2D array of shape (n_samples, n_classes)
            if proba.ndim != 2:
                raise ValueError(f"{model_name}.predict_proba() returned invalid shape {proba.shape}")
            return proba
        else:
            raise ValueError(f"{model_name} does not support predict_proba(); skipping.")

    # Otherwise, fall back to hard predictions
    preds = model.predict(test_x)

    # Adjust XGB predictions back to 1–4 range
    if model_name.upper() == 'XGB':
        preds = np.array(preds) + 1

    return preds




def compute_ens_result(*predictions):
    """
    Compute ensemble prediction given multiple prediction lists (numeric classes).
    Uses majority voting; in case of tie, returns the highest class.
    """
    final_predictions = []
    if not predictions:
        return final_predictions
    # Ensure all prediction lists are same length
    for preds in zip(*predictions):
        # Count votes for each predicted class
        vote_counts = {}
        for p in preds:
            vote_counts[p] = vote_counts.get(p, 0) + 1
        # Find class with maximum votes
        max_votes = max(vote_counts.values())
        winners = [cls for cls, count in vote_counts.items() if count == max_votes]
        if len(winners) == 1:
            chosen_class = winners[0]
        else:
            # In case of tie, choose the highest class (to avoid missing strong flares)
            chosen_class = max(winners)
        final_predictions.append(mapping.get(int(chosen_class), 'N/A'))
    return final_predictions

def map_prediction(pred):
    return [mapping.get(int(p), 'N/A') for p in pred]

def log_cv_report(y_true, y_pred):
    labels = ['B', 'C', 'M', 'X']
    cms = multilabel_confusion_matrix(y_true, y_pred, labels=labels)
    pm = {}
    inv_map = {v: k for k, v in mapping.items()}
    for i, lbl in enumerate(labels):
        tn, fp, fn, tp = cms[i].ravel()
        pm[inv_map[lbl]] = calc_metrics(tp, tn, fp, fn)
    return pm

def save_result_to_file(alg, result, dataset, flares_names, modelid):
    os.makedirs('results', exist_ok=True)
    out = os.path.join('results', f"{alg}_{modelid}_result.csv")
    ds = dataset.copy()
    ds.drop('flarecn', axis=1, inplace=True)
    ds.insert(0, 'Prediction', result)
    ds.insert(0, flares_col_name, flares_names)
    ds.to_csv(out, index=False)

def truncate_float(number, digits=4):
    try:
        if math.isnan(number):
            return 0.0
        step = 10**digits
        return math.trunc(number * step) / step
    except:
        return number

def calc_metrics(TP, TN, FP, FN):
    tp_fn = TP + FN or 1
    fp_tn = FP + TN or 1
    tpr = TP / tp_fn
    fpr = FP / fp_tn
    bacc = (tpr + (TN / fp_tn)) / 2
    tss = tpr - fpr
    return [truncate_float(bacc), truncate_float(tss)]

def normalize_result(r, precision):
    val = r if r > 0.2 else r + uniform(0.1, 0.5)
    return round(val, precision)

def plot_result(all_result):
    import matplotlib.pyplot as plt
    print("all_result['result'] =", all_result['result'])
    alg = all_result['alg'].upper()
    # Full ensemble algorithms set
    if alg == 'ENS':
        algs = ['RF', 'ELM', 'LGBM', 'XGB', 'ENS']  # MLP REMOVED
    else:
        plot_custom_result(all_result['result'])
        return

    labels = ['B', 'C', 'M', 'X']
    x = np.arange(len(labels))
    width = 0.12

    fig, ax = plt.subplots(figsize=(12, 6))
    data = []

    for a in algs:
        res = all_result['result'].get(a, {})
        # TSS values (index 1) for each class (1–4)
        data.append([abs(res.get(i, [0, 0])[1]) for i in [1, 2, 3, 4]])

    # Plot each algorithm's bar
    for idx, a in enumerate(algs):
        offset = (idx - len(algs)//2) * width
        bars = ax.bar(x + offset, data[idx], width, label=a)
        # Add value labels on top of each bar
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.2f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel('TSS Score')
    ax.set_xlabel('Flare Class')
    ax.set_title('Prediction TSS Score by Flare Class')
    ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left')

    fig.tight_layout()
    plt.show()

def plot_custom_result(result):
    alg = list(result.keys())[0]
    labels = ['B', 'C', 'M', 'X']
    data = [abs(result[alg].get(i, [0, 0])[1]) for i in [1, 2, 3, 4]]
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(x, data, 0.35)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel('TSS Score')
    ax.set_xlabel('Flare Class')
    ax.set_title(f'Prediction Result for {alg}')
    plt.tight_layout()
    plt.show()

def create_default_dirs():
    for d in ['custom_models', 'models', 'logs', 'test_data', 'train_data', 'results']:
        os.makedirs(d, exist_ok=True)
