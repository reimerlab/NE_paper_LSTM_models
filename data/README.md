# Data

## Dataset access

The full dataset underlying this study is available at:

> https://dandiarchive.org/dandiset/001865

## Expected directory structure

After downloading, organize files as follows:
data/
├── hemodynamic_prediction/
│ ├── rNE-mut_prediction_data/
│ │ ├── M1_12345_3_1.csv
│ │ └── ...
│ └── rNE-BrightnessBins/
│ ├── M1_12345_3_1_BrightnessBins.csv
│ └── ...
├── rNE0.5/
│ ├── 12345_3_1.csv
│ └── ...
├── GrabNE2h/
│ ├── 12346_2_1.csv
│ └── ...
└── README.md

Files for the hemodynamic correction model follow the naming convention
`{brain_area}_{animal_id}_{session}_{scan_idx}`. Files for the behavioral
prediction model follow the convention `{animal_id}_{session}_{scan_idx}`,
organized into subfolders by sensor type (`rNE0.5/` or `GrabNE2h/`).

## Main recording CSV columns

| Column | Units | Description |
|--------|-------|-------------|
| `fluorescence_timestamps` | s | Frame times for the fluorescence recording |
| `rNE_fluorescence` | a.u. | Raw rNE0.5 fluorescence trace |
| `mut_fluorescence` | a.u. | Raw inert control (mut) fluorescence trace |
| `velocity_in_cm_per_sec` | cm/s | Treadmill speed (raw) |
| `velocity_timestamps` | s | Treadmill timestamps |
| `radius_in_pixels` | px | Pupil radius (raw) |
| `radius_timestamps` | s | Pupil camera timestamps |

## Brightness-bin CSV columns

| Column | Description |
|--------|-------------|
| `timestamps` | Timestamps (s) |
| `Mask_1` … `Mask_10` | Mean fluorescence in each of 10 spatial brightness bins |
