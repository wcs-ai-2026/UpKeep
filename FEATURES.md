- `wind_speed_avg`:  average wind speed.
- `grid_power_avg`: average grid power. Grid power had already been normalized
- `ambient_temp`:  outside temperature.
- `gust_factor`: `wind_speed_max / wind_speed_avg`; missing when average is zero.
- `turbulence`: `wind_speed_std / wind_speed_avg`; missing when average is zero.
- `wind_jump`: Current `wind_speed_avg` minus previous reading's average.
- `gust_events`: `1` when `gust_factor >= 1.5`, otherwise `0`; missing when gust factor is missing; temporary variable.
- `gusts_last_24h`: Sum of `gust_events` in the previous 24 hours, excluding the current reading; empty windows give `0`.
- `gusts_last_7d`: Sum of `gust_events` in the previous 7 days, excluding the current reading; empty windows give `0`.
- `temp_1h_ago`: `gearbox_bearing_temp.shift(6)`; missing unless six readings span exactly one hour.
- `hour_of_day`: `timestamp.dt.hour`, from `0` to `23`.
- `generator_bearing_1_temp_1h_ago`: `generator_bearing_1_temp.shift(6)`.
- `generator_bearing_2_temp_1h_ago`: `generator_bearing_2_temp.shift(6)`.

Each generator lag requires the sixth previous row to be exactly one hour earlier and to contain that bearing's temperature.

Run `python -m src.upkeep.features 0 10 25` from the repository root to build feature Parquet files in `data/processed/` from the corresponding cleaned CSVs.
