# FlareBlueML (FlareML v1.3)

## Overview

FlareBlueML (formerly FlareML v1.x) is a Python-based framework for multiclass solar flare prediction, leveraging magnetic field features from solar active regions to forecast B-, C-, M-, and X-class flares. This project enhances the original FlareML toolkit by integrating advanced temporal feature engineering, consistent data normalization, robust time-series cross-validation, and a probability‑averaging ensemble of four base learners: Random Forest (RF), Extreme Learning Machine (ELM), LightGBM (LGBM), and XGBoost (XGB).

Key features:

* **Temporal Feature Engineering**: Rolling statistics, anomaly scores, velocity, and acceleration for selected SHARP parameters.
* **Model Suite**: RF, ELM (with saved scaler), LGBM (with balanced weighting), XGB (with label alignment).
* **Ensemble Strategy**: Soft probability averaging with tie-break preference for higher classes.
* **Reproducible Pipeline**: Train/test scripts (`flareml_train.py`, `flareml_test.py`), saved normalization and scaler parameters, detailed logging, and result exports.

## Repository Structure

```
├── flareml_utils.py     # Core utilities: preprocessing, feature engineering, model wrappers
├── flareml_train.py     # Training script with optional normalization and hyperparameter tuning
├── flareml_test.py      # Testing/evaluation script with feature preview and ensemble prediction
├── data/                # Data directory (train_data/ and test_data/ subfolders)
├── custom_models/       # Saved models for non-default model IDs
├── models/              # Default models directory
├── logs/                # Log files
├── results/             # CSV results and plots (e.g., performance metrics)
├── requirements.txt     # Python dependencies
└── README.md            # This file
```

## Requirements

* Python 3.8+
* Libraries (see `requirements.txt`):

  * numpy
  * pandas
  * scikit-learn
  * xgboost
  * lightgbm
  * sklearn-elm (for ELM)
  * matplotlib

To install dependencies:

```bash
pip install -r requirements.txt
```

## Data Preparation

* Place your training data CSV in `data/train_data/`, e.g., `flaringar_original_data.csv`.
* Place your test data CSV in `data/test_data/`, e.g., `flaringar_simple_random_40.csv`.
* Ensure the dataset contains the `fdate`, `Flare Class`, and SHARP feature columns (e.g., `TOTUSJH`, `TOTBSQ`, `SAVNCPP`, `R_VALUE`).

## Training

Run the training script to fit models and save outputs:

```bash
python flareml_train.py \
  -a ENS \            # Algorithm: ENS, RF, ELM, LGBM, or XGB
  -m my_model_id \    # Model identifier (folder prefix)
  -n True \           # Enable global min-max normalization (optional)
  -v True             # Verbose logging
```

* **--normalize\_data**: When set to `True`, scales all features to \[0,1] and saves min/max as `my_model_id_norm_params.json`.
* The script sequentially trains each selected model and saves:

  * `<model_id>_rf.sav`, `<model_id>_elm.sav`, etc., in `models/` or `custom_models/`.
  * For ELM, also saves `<model_id>_elm_scaler.sav`.

## Testing & Evaluation

Evaluate on a held-out test set:

```bash
python flareml_test.py \
  -a ENS \            # Must match training algorithm (ENS for ensemble)
  -m my_model_id \    # Model identifier
  -n True \           # If normalization was used at train time
  -v True             # Verbose logging
```

* Prints a **Detailed Feature Engineering Preview** of the first 10 rows and summary statistics.
* Applies the same preprocessing and normalization saved during training.
* Generates predictions for each model and the ensemble (probability averaging).
* Displays confusion matrices and prints per-class Balanced Accuracy (BACC) and True Skill Statistic (TSS).
* Plots TSS bar chart via `plot_result()`.

## Configuration & Hyperparameters

* **Temporal window**: Default 6 data points (hours) for rolling features; configurable via `engineer_temporal_features(window_size=...)`.
* **Random Forest**: `n_estimators=200`, `class_weight='balanced'`, tuneable in `rf_train_model()`.
* **LightGBM / XGBoost**: Grid search over `n_estimators`, `learning_rate`, `max_depth`/`num_leaves`, and subsampling; uses `TimeSeriesSplit`.
* **ELM**: 200 hidden neurons (tanh activation); scales with `StandardScaler`.
* **Ensemble**: Probability averaging across available models; requires at least two probability-capable models.

## Logging & Outputs

* **logs/**: Contains `ens_deepsun.log` with training/testing details.
* **results/**: CSV files like `<alg>_<model_id>_result.csv` listing each test sample’s true and predicted classes.
* **Plots**: TSS by class chart displayed and can be saved manually via the plotting window.

## Contributing

1. Fork the repository and create a new branch.
2. Add or modify features (e.g., new model, different features).
3. Ensure compatibility with existing pipeline and add tests.
4. Submit a pull request for review.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

*This README was generated for FlareBlueML (FlareML v1.3) by Daniel Graves.*
