# Config field reference

Use this reference with [default.yaml](../configs/real_data/chen2024/default.yaml).

## Top-level fields

- `input_dir` (default: `raw`)  
  Path to NWB file or directory of NWB files.

- `output_dir` (default: `processed`)  
  Output folder for generated `sub-*.h5` files.

- `subjects` (default: `[]`)  
  Optional list to restrict processing to specific subjects.
  - `id`: subject identifier, with or without `sub-` prefix.
  - `name`: human-readable label written to output as `subject_name`.
  - With an empty list, all subjects in `input_dir` are processed.

## `trial_filter`

- `duration_tolerance_s` (default: `0.5`)  
  Accepted absolute deviation from the fixed experimental trial duration of 5.0 seconds.  
  Trials are kept when `abs((stop_time - start_time) - 5.0) <= duration_tolerance_s`.

- `photostim` (default: `exclude`)  
  One of:
  - `exclude`: keep only non-stim trials
  - `include`: keep both stim and non-stim trials
  - `only`: keep only stim trials

  A trial counts as non-stim only if its `photostim_onset` is exactly `"N/A"` (the rule of the original extraction); any other value counts as stimulated.

- Only accepted trials are written. `source_trial_row` records each written trial's row in the NWB trial table (which equals its row in the original extraction, since that file kept every trial).

## `binning`

- `mode` (default: `fixed_bins`)  
  Current pipeline supports fixed-number-of-bins behavior per accepted trial.  
  `fixed_bins` is currently the only supported mode.

- `n_bins` (default: `500`)  
  Number of bins created inside each accepted trial window `[start_time, stop_time)`.

### `fixed_bins` and `duration_tolerance_s` relationship

With `fixed_bins`, bin count is constant but bin width is per-trial:

- `bin_width_s = trial_duration_s / n_bins`
- shorter accepted trials -> smaller bins
- longer accepted trials -> larger bins

So `duration_tolerance_s` controls how much bin-width variability is allowed:

- smaller tolerance (for example `0.1`) -> durations are closer to 5.0 s -> more consistent bin widths
- larger tolerance (for example `0.5`) -> wider duration spread -> more bin-width variability

The output always stores per-trial `trial_duration_s` and `bin_width_s` as info datasets.

## `units`

- `classification` (default: `good`)  
  Keep only units with that classification value.  
  Set `null` to keep all units.

## `metadata.trial_fields`

- Mapping from NWB trial column -> output dataset name.
- Example: `photostim_onset: stim_onset` writes dataset `stim_onset`.
- If omitted, scalar trial columns are copied with original names.
- If a configured source field is missing in a file, preprocessing fails explicitly.

Common defaults from [default.yaml](../configs/real_data/chen2024/default.yaml):

- `start_time`, `stop_time`, `photostim_onset`, `photostim_duration`, `trial_uid`, `outcome`, `trial_instruction`
- The benchmark's real-data datasets also keep `outcome` and `trial_instruction` (both in `example_440959.yaml`).

## `region_labels`

- `abbreviation` (default): each unit's `anno_name` (a full Allen name in DANDI 000363, e.g. "Secondary motor area, layer 2/3") is translated to its Allen CCF v3 abbreviation (`MOs2/3`) with the packaged table `src/chen2024/data/allen_ccf_v3_names.csv`. This is what the original extraction did. Names missing from the table are kept as they are, with a warning.
- `full_name`: keep `anno_name` unchanged.

## `regions`

- Default: `{}` (empty mapping)  
  Keeps every area as its own `area-*` dataset (one per abbreviation; `/` becomes `_`, so `MOs2/3` is written as `area-MOs2_3`).

- Mapping mode:
  - groups listed areas under your custom output labels (one `area-<group>` dataset each)
  - list abbreviations (or full Allen names); matching is exact
  - unlisted areas are excluded

## Included examples

- [example_440959.yaml](../configs/real_data/chen2024/example_440959.yaml): paper-style 440959 grouping.
- [example_single_subject.yaml](../configs/real_data/chen2024/example_single_subject.yaml): minimal single-subject convenience preset.
