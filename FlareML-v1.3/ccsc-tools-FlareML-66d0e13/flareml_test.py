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

import os
import sys
import argparse
import numpy as np
from sklearn.metrics import confusion_matrix

from flareml_utils import (
    load_dataset_csv,
    removeDataColumn,
    convert_class_to_num,
    normalize_scale_data,
    engineer_temporal_features,
    map_prediction,
    log_cv_report,
    are_model_files_exist,
    get_partial_ens_trained,
    model_prediction_wrapper,
    default_models_dir,
    custom_models_dir,
    flares_col_name,
    algorithms,
    boolean,
    set_log_to_terminal,
    log,
    compute_ens_result,
    plot_result,
    mapping
)

# Default test data
TEST_INPUT = 'data/test_data/flaringar_simple_random_40.csv'
normalize_data = False


def safe_model_predict(name, test_x, test_y, model_id):
    key = name.upper()
    models_dir = custom_models_dir if model_id != 'default_model' else default_models_dir
    if not are_model_files_exist(models_dir, model_id, alg=key):
        print(f"Model file for {key} not found; skipping.")
        return None
    try:
        print(f"Predicting with {key}...")
        preds = model_prediction_wrapper(key, None, test_x, test_y, model_id=model_id)
        return preds
    except Exception as e:
        print(f"Error during {key} prediction: {e}")
        return None


def test_model(args):
    alg = args.get('algorithm', 'ENS').strip().upper()
    if alg not in algorithms:
        print(f"Invalid algorithm: {alg}. Must be one of: {algorithms}")
        sys.exit(1)

    test_file = args.get('test_data_file', TEST_INPUT).strip()
    if not test_file or not os.path.isfile(test_file):
        print(f"Testing data file is invalid or not found: {test_file}")
        sys.exit(1)

    model_id = args.get('modelid', 'default_model').strip()
    if not model_id:
        print("Model id cannot be empty.")
        sys.exit(1)

    verbose = boolean(args.get('verbose', False))
    set_log_to_terminal(verbose)

    # Check model availability
    models_dir = custom_models_dir if model_id != 'default_model' else default_models_dir
    exists = are_model_files_exist(models_dir, model_id, alg=alg)
    partial = get_partial_ens_trained()
    if alg == 'ENS' and not exists and not partial:
        print(f"Model id {model_id} does not exist for algorithm {alg}. Please train first.")
        sys.exit(1)

    print(f"Loading test data: {test_file}")
    df = load_dataset_csv(test_file)

    # Feature engineering
    df = engineer_temporal_features(df, window_size=6)
    
     # after df = engineer_temporal_features(df, window_size=6)

# ————— Detailed Feature‑Engineering Preview —————
    features = ['TOTUSJH','TOTBSQ','SAVNCPP','R_VALUE']  
    all_cols = []
    for feat in features:
        all_cols += [
            feat, 
            f"{feat}_roll_mean", 
            f"{feat}_roll_std", 
            f"{feat}_roll_skew", 
            f"{feat}_anomaly", 
            f"{feat}_velocity", 
            f"{feat}_acceleration"
    ]

# Filter only the columns that actually exist 
    preview_cols = [c for c in all_cols if c in df.columns]

    print("\n=== Detailed Feature Engineering Preview (first 10 rows) ===")
    if preview_cols:
    # prints a nicely formatted table of the first 10 rows
        print(df[preview_cols].head(10).to_string(index=False))
    else:
        print("No engineered temporal features found.")

        print("\n=== Summary Statistics for Engineered Features ===")
# for each col, print mean, std, min, max
    for c in preview_cols:
        col = df[c]
        print(f"{c:30s} | mean={col.mean():8.3f}  std={col.std():8.3f}  "
                        f"min={col.min():8.3f}  max={col.max():8.3f}")
        print("=== End of Feature Engineering Preview ===\n")
    
    for col in ['goes', 'fdate', 'goesstime', 'flarec', 'noaaar']:
        df = removeDataColumn(col, df)

    if flares_col_name not in df.columns:
        print(f"Missing target column '{flares_col_name}' in test data.")
        sys.exit(1)

    # Apply normalization if specified (uses test's min-max)
    if boolean(args.get('normalize_data', False)):
        import json
        norm_path = f"{model_id}_norm_params.json"
        with open(norm_path, 'r') as jf:
            norm = json.load(jf)
        for c in df.columns:
            if c != flares_col_name:
                mn, mx = norm['min'][c], norm['max'][c]
                arr = df[c].values
                df[c] = (arr - mn) / (mx - mn or 1)
        print(f"[INFO] Loaded normalization params from {norm_path}")


    # Prepare target
    df['flarecn'] = [convert_class_to_num(c) for c in df[flares_col_name]]
    df = removeDataColumn(flares_col_name, df)
    test_y = df['flarecn']
    test_x = removeDataColumn('flarecn', df)
    true_y = [mapping.get(y, 'N/A') for y in test_y]

    # Generate predictions
    rf_res = elm_res = lgbm_res = xgb_res = None
    if alg in ['RF', 'ENS']:
        rf_res = safe_model_predict('RF', test_x, test_y, model_id)
    if alg in ['ELM', 'ENS']:
        elm_res = safe_model_predict('ELM', test_x, test_y, model_id)
    if alg in ['LGBM', 'ENS']:
        lgbm_res = safe_model_predict('LGBM', test_x, test_y, model_id)
    if alg in ['XGB', 'ENS']:
        xgb_res = safe_model_predict('XGB', test_x, test_y, model_id)

    print("\nAvailable model predictions:")
    for m, res in [('RF', rf_res), ('ELM', elm_res), ('LGBM', lgbm_res), ('XGB', xgb_res)]:
        print(f"{m}: {'✓' if res is not None else '✗'}")

    performance = {}
    # Ensemble evaluation
        # Ensemble evaluation
    if alg == 'ENS':
        # ——— Probability‑averaging ensemble ———
        probas = {}
        for name, res in [('RF', rf_res), ('ELM', elm_res),
                          ('LGBM', lgbm_res), ('XGB', xgb_res)]:
            if res is not None:
                try:
                    p = model_prediction_wrapper(
                        name, None, test_x, test_y, model_id,
                        return_proba=True
                    )
                    probas[name] = p  # shape: (n_samples, n_classes)
                except Exception as e:
                    print(f"Skipping {name} probabilities: {e}")

        if len(probas) >= 2:
            print("Computing ensemble (probability averaging)...")
            import numpy as np
            # stack and average across models
            stacked   = np.stack(list(probas.values()), axis=0)
            avg_proba = stacked.mean(axis=0)  # (n_samples, n_classes)
            # pick the class with highest mean probability
            idx_preds = np.argmax(avg_proba, axis=1)  # 0–3
            num_preds = idx_preds + 1                 # shift to 1–4
            ens_preds = [mapping[n] for n in num_preds]
            performance['ENS'] = log_cv_report(true_y, ens_preds)
            print("\nConfusion Matrix for ENS (prob avg):")
            print(confusion_matrix(true_y, ens_preds, labels=['B','C','M','X']))
        else:
            print("Ensemble could not be computed (need ≥2 models with probabilities).")

        # Log individual model performance
        for m, res in [('RF', rf_res), ('ELM', elm_res), ('LGBM', lgbm_res), ('XGB', xgb_res)]:
            if res is not None:
                mapped = map_prediction(res)
                print(f"\nConfusion Matrix for {m}:")
                print(confusion_matrix(true_y, mapped, labels=['B','C','M','X']))
                performance[m] = log_cv_report(true_y, mapped)
    else:
        # Single-model evaluation
        res = locals().get(f"{alg.lower()}_res")
        if res is not None:
            mapped = map_prediction(res)
            print(f"\nConfusion Matrix for {alg}:")
            print(confusion_matrix(true_y, mapped, labels=['B','C','M','X']))
            performance[alg] = log_cv_report(true_y, mapped)
        else:
            print(f"No predictions for {alg}; nothing to report.")

    # Summary of metrics
    print("\n=== Performance Metrics ===")
    for name, metrics in performance.items():
        print(f"-- {name} --")
        for cls, vals in metrics.items():
            print(f"Class {cls}: BACC={vals[0]}, TSS={vals[1]}")

    return {'alg': alg, 'result': performance}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--test_data_file', default=TEST_INPUT)
    parser.add_argument('-a', '--algorithm', default='ENS')
    parser.add_argument('-m', '--modelid', default='default_model')
    parser.add_argument('-v', '--verbose', default=False)
    parser.add_argument('-n', '--normalize_data', default=normalize_data)
    args = vars(parser.parse_args())
    results = test_model(args)
    plot_result(results)
