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
import argparse
import sys
import time
from flareml_utils import (
    load_dataset_csv,
    removeDataColumn,
    convert_class_to_num,
    normalize_scale_data,
    rf_train_model,
    elm_train_model,
    lgbm_train_model,
    xgb_train_model,
    create_default_dirs,
    boolean,
    set_log_to_terminal,
    log,
    default_models_dir,
    custom_models_dir,
    engineer_temporal_features,
    flares_col_name,
    algorithms
)

# Ensure required directories exist
create_default_dirs()

TRAIN_INPUT = 'data/train_data/flaringar_original_data.csv'
normalize_data = False


def train_model(args):
    alg = args.get('algorithm', 'ENS').strip().upper()
    # Validate algorithm choice
    if alg not in algorithms:
        print(f"Invalid algorithm: {alg}. Must be one of: {algorithms}")
        sys.exit(1)

    train_file = args.get('train_data_file', TRAIN_INPUT).strip()
    if not train_file or not os.path.isfile(train_file):
        print(f"Training data file invalid or not found: {train_file}")
        sys.exit(1)

    modelid = args.get('modelid', 'default_model').strip()
    if not modelid:
        print("Model id cannot be empty.")
        sys.exit(1)
    if modelid.lower() == 'default_model':
        ans = input('Using default_model will overwrite defaults. Continue? [y/N] ')
        if not boolean(ans):
            print('Exiting..')
            sys.exit(1)

    normalize = boolean(args.get('normalize_data', False))
    verbose = boolean(args.get('verbose', False))
    set_log_to_terminal(verbose)
    log('Arguments:', args)
    print(f"Loading training data: {train_file}")
    df = load_dataset_csv(train_file)

    # Ensure temporal feature engineering is possible
    if 'fdate' not in df.columns:
        print("Error: Training dataset has no 'fdate' column. Cannot engineer temporal features.")
        sys.exit(1)

    # Feature engineering
    df = engineer_temporal_features(df, window_size=6)

    # Drop unused columns
    for col in ['goes', 'fdate', 'goesstime', 'flarec', 'noaaar']:
        df = removeDataColumn(col, df)

    if flares_col_name not in df.columns:
        print(f"Missing target column '{flares_col_name}' in dataset.")
        sys.exit(1)

    # Convert target to numeric and drop original
    df['flarecn'] = [convert_class_to_num(x) for x in df[flares_col_name]]
    df = removeDataColumn(flares_col_name, df)

    if normalize:
        mins, maxs = {}, {}
        for c in df.columns:
            if c != 'flarecn':
                arr = df[c].values
                mins[c], maxs[c] = arr.min(), arr.max()
                # scale to [0,1]
                df[c] = (arr - mins[c]) / (maxs[c] - mins[c] or 1)
        # save these parameters so we can reuse them at test time
        import json
        norm_path = f"{modelid}_norm_params.json"
        with open(norm_path, 'w') as jf:
            json.dump({'min': mins, 'max': maxs}, jf)
        print(f"[INFO] Saved normalization params to {norm_path}")

    train_y = df['flarecn']
    train_x = removeDataColumn('flarecn', df)
    test_x = test_y = None

    print('Training in progress...')
    if alg == 'ENS':
        # Train all base models for ensemble
        rf_train_model(train_x, test_x, train_y, test_y, model_id=modelid)
        print('Finished RF training.')
        elm_train_model(train_x, test_x, train_y, test_y, model_id=modelid)
        print('Finished ELM training.')
        lgbm_train_model(train_x, test_x, train_y, test_y, model_id=modelid)
        print('Finished LGBM training.')
        xgb_train_model(train_x, test_x, train_y, test_y, model_id=modelid)
        print('Finished XGB training.')
    elif alg == 'RF':
        rf_train_model(train_x, test_x, train_y, test_y, model_id=modelid)
    elif alg == 'ELM':
        elm_train_model(train_x, test_x, train_y, test_y, model_id=modelid)
    elif alg == 'LGBM':
        lgbm_train_model(train_x, test_x, train_y, test_y, model_id=modelid)
    elif alg == 'XGB':
        xgb_train_model(train_x, test_x, train_y, test_y, model_id=modelid)
    else:
        print(f"Unsupported algorithm: {alg}")
        sys.exit(1)

    print(f"Training complete for algorithm {alg}. You may now run flareml_test.py.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--train_data_file', default=TRAIN_INPUT)
    parser.add_argument('-v', '--verbose', default=False)
    parser.add_argument('-a', '--algorithm', default='ENS')
    parser.add_argument('-m', '--modelid', default='default_model')
    parser.add_argument('-n', '--normalize_data', default=normalize_data)
    args = vars(parser.parse_args())
    train_model(args)
