# Decisions

- Nov 15, 2026 (Varun, SCRUM-36): Tuned LightGBM settings on validation only
  (last 2 months of each dataset's train split, per the plan - never touched
  the locked test/prediction rows). Script: tune_lgbm_settings.py, full
  results in tuning_results.csv (60 rows: 4 models x 3 datasets x 5 settings).

  Chosen: n_estimators=300, learning_rate=0.05 (up from the old default of
  200 estimators). Won or tied on 3 of 4 models; the 4th (generator_bearing_1)
  was only ~0.04 degC worse, within noise. Validation MAE, averaged across
  datasets 0/10/25, old default vs chosen:
  - gearbox_bearing_temp: 0.824 -> 0.817
  - generator_bearing_1_temp: 3.077 -> 3.122
  - generator_bearing_2_temp: 3.438 -> 3.436
  - gearbox_oil_temp: 1.199 -> 1.180

  Note: the generator bearing models are still above the project's 2 degC
  target even after tuning - tuning the settings barely moved these numbers,
  so the bottleneck isn't the settings. Likely needs a better feature (e.g.
  a generator-specific "temp 1h ago" instead of the gearbox-based one
  currently in FEATURES.md) rather than more tuning. Flagging for the team,
  not fixing in this ticket.

- Nov 15, 2026 (Varun): Pinned requirements.txt to the exact library versions
  tested (lightgbm 4.7.0, pandas 2.3.3, numpy 2.2.6, scikit-learn 1.7.2,
  pyarrow 25.0.1, joblib 1.6.0), and added random_state=42 to the LightGBM
  model so results are reproducible run to run on the same machine. Small
  differences can still happen across different machines/thread counts -
  pinning gets everyone close, not byte-for-byte identical.

- Nov 15, 2026 (Varun): Training scripts now drop any row with a missing
  target value before fitting or predicting (found 1 missing
  generator_bearing_1_temp reading in datasets 0 and 10) instead of silently
  including it. Predictions CSVs now include a "split" column (train vs.
  prediction) so train and test rows can be told apart. Removed the
  held-out MAE printing from the training scripts - evaluation is
  run_eval.py's job, not the training script's.

- Nov 24, 2026 (Varun, lock final models): Locked the final model versions
  for the summit. All saved models uploaded to the shared Drive folder
  (models_saved/, which stays out of git on purpose).

  LightGBM settings: n_estimators=300, learning_rate=0.05, random_state=42
  (the Nov 15 tuning decision above), trained with the generator-specific
  lag feature fix (generator_bearing_1_temp_1h_ago,
  generator_bearing_2_temp_1h_ago) from SCRUM-35.

  Isolation Forest settings: n_estimators=200, random_state=42, scores
  calibrated to a 0 to 1 range using the 1st and 99th percentile of the
  training residuals.

  15 files total, covering datasets 0, 10 and 25: 3 gearbox bearing, 3
  gearbox oil (bonus), 3 generator bearing 1, 3 generator bearing 2, 3
  Isolation Forest.
