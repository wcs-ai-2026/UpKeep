# Decisions

- Nov 15, 2026 (Varun): Tried a few different settings for the LightGBM models
  (more/fewer trees, different learning rates) and tested them the right way -
  on the last 2 months of training data only, never touching the locked test
  period. Honestly the difference between settings was pretty small either way
  (under half a degree). Went with 300 trees instead of the old default of 200,
  since it came out best or close to best on 3 of the 4 models. Updated this
  in src/upkeep/models/lgbm_nbm.py.
