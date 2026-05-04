# Wearable EEG Attention Tracking across Dynamic Listening Scenarios

![Visual Abstract](article/figures/visualAbstract.pdf)

Everyday communication is dynamic and multisensory — involving shifting attention, overlapping speech and visual cues. Yet most neural attention tracking studies are still limited to highly controlled lab settings. This work introduces a novel dataset from 24 normal-hearing participants recorded with a wearable EEG system (44 scalp electrodes + 20 cEEGrid in-ear electrodes) across three audiovisual conditions: **sustained attention** to a single talker in a two-talker environment, **attention switching** between two talkers, and **unscripted two-talker conversations** with a competing side talker.

---

## Repository structure

```
.
├── data/                        # Raw and derived EEG data (BIDS-like layout)
│   ├── sub-{id}/
│   │   └── eeg/
│   │       └── sub-{id}_task-{cond}_eeg.xdf          # raw XDF recording
│   └── derivatives/
│       ├── montage.sfp                  # electrode position file
│       ├── behav/sub-{id}/              # randomization CSVs (per condition)
│       ├── ica/                         # fitted ICA solutions (.fif)
│       ├── logs/                        # per-subject preprocessing logs (CSV)
│       └── preprocessed/sub-{id}/      # final epoch + predictor bundles (.pickle)
│
├── stimuli/                     # Stimulus audio files (.wav / .mp3)
├── predictors/                  # Derived gammatone predictor files (.pickle)
├── results/                     # Fitted TRF model bundles (.pkl)
│   ├── sustA.pkl                # analysis-ready TRF dicts
│   ├── switA.pkl
│   └── convA.pkl
│
├── figures/                     # Generated figures
├── article/figures/             # Manuscript figures
│
├── preprocessing.ipynb          # Step 1 — EEG preprocessing & predictor generation
├── runTRFs.ipynb                # Step 2 — TRF model fitting
├── analysis.ipynb               # Step 3 — group analysis & figure generation
│
├── experiment.py                # MobEEG TRFExperiment class (eelbrain pipeline)
├── helpers.py                   # XDF loading, gammatone computation, epoch utilities
├── plotting.py                  # All visualisation functions
├── select_ica_gui.py            # Standalone ICA selection GUI (subprocess helper)
└── trf.yml                      # Conda environment specification
```

---

## Data availability

<!--
One subject (`sub-99`) is provided as a worked example so that the full `preprocessing.ipynb` pipeline can be run end-to-end:

- `data/sub-99/eeg/` — raw XDF recording and converted FIF files for the `sustA` condition.
- `data/derivatives/behav/sub-99/` — three randomization CSVs (one per condition).
- `data/derivatives/ica/` — where ICA solutions is stored `sub-99`.
-->

The data underlying this study are subject to ethical and legal restrictions. Participant consent did not explicitly include permission for public data sharing, and the dataset falls under the scope of the EU General Data Protection Regulation (GDPR). For these reasons, the data are not publicly available. Data may be made available upon reasonable request, subject to institutional and ethical approval and appropriate data‑sharing agreements.

**Pre-fitted TRF results** for all participants are provided in `results/` (`sustA.pkl`, `switA.pkl`, `convA.pkl`) so that `analysis.ipynb` can be run immediately to reproduce all manuscript figures without re-running the preprocessing or TRF fitting steps.

---

## Installation

1. Install [Miniconda](https://docs.conda.io/en/latest/miniconda.html) or Anaconda.

2. Create and activate the environment:

   ```bash
   conda env create -f trf.yml
   conda activate trf
   ```

The environment installs **eelbrain ≥ 0.42.0a4**, MNE-Python, and all other dependencies. If you prefer pip, install the packages listed in `trf.yml` manually; eelbrain requires `wxPython` for the GUI cells in `preprocessing.ipynb`.

---

## Data structure and BIDS conventions

The pipeline follows a BIDS-inspired layout. Raw EEG recordings are stored as `.xdf` files (from Lab Streaming Layer) under `data/sub-{id}/eeg/`. The preprocessing notebook converts these to MNE `.fif` format and writes both a task-agnostic copy (`sub-{id}_acq-{acq}_eeg.fif`) and a BIDS-compliant copy with the `task-` field (`sub-{id}_task-{cond}_acq-{acq}_eeg.fif`), which is required by eelbrain's TRF pipeline.

**Acquisition labels** encode both the sensor array and condition in a single field (e.g. `scalpSustA`, `ceegridSwitA`). This is necessary because eelbrain's pipeline uses the `acquisition` field to route data through condition-specific ICA states.

**Eelbrain TRF pipeline** — the `MobEEG` class in `experiment.py` subclasses `trftools.TRFExperiment`, which automates filter, ICA, epoch, and TRF-fitting steps. See the eelbrain publication for conceptual background:

> Brodbeck, C. et al. (2023). *Eelbrain, a Python toolkit for time-continuous analysis with temporal response functions.* eLife 12:e85012. <https://elifesciences.org/articles/85012>

and the trftools pipeline documentation for the `TRFExperiment` API:

> <https://trf-tools.readthedocs.io/latest/pipeline.html>

**Behavioural randomization files** (`data/derivatives/behav/sub-{id}/`) map each trial to its attended and ignored stimulus files:

| File | Condition | Columns |
|------|-----------|---------|
| `randomization1.csv` | sustA | `AttendedFile`, `IgnoredFile`, `TrialNumber` |
| `randomization2.csv` | switA | `AttendedFile`, `IgnoredFile`, `TrialNumber` |
| `randomization3.csv` | convA | `AttendedFile`, `IgnoredFile`, `TrialNumber` |

To add a new subject, place the XDF recording under `data/sub-{id}/eeg/` and the corresponding randomization CSVs under `data/derivatives/behav/sub-{id}/`, then run the three notebooks in order.

---

## Scripts

### `experiment.py`
Defines the `MobEEG` class (subclass of `trftools.TRFExperiment`) and all preprocessing helpers.

- **`MobEEG`** — manages the eelbrain pipeline graph: raw → band-pass filter → ICA fit → ICA apply → final band. Implements `align_epochs_and_predictors()`, the main method that loads EEG epochs, applies the ICA solution, resamples to 50 Hz, z-scores each channel, and time-aligns the gammatone predictor NDVars. Handles the switA attention-switch splitting logic internally.
- **`_build_raw_pipelines()`** — constructs the dictionary of `RawPipeline` steps for all sensor arrays and task conditions.
- **Logging helpers** (`_log_append`, `_save_log_csv`, `_preprocess_pred_with_log`) — record every filter, resample, and z-score step to per-subject CSV files in `data/derivatives/logs/`.

### `helpers.py`
Utility functions for loading raw data and computing stimulus features.

- **`save_xdf_as_fif()`** — loads an XDF recording via `pyxdf`, applies manual clock synchronisation, builds an MNE montage from `montage.sfp`, applies notch filtering, splits into scalp and cEEGrid datasets, and saves `.fif` files and event files.
- **`make_gammatone()`** — computes and caches high-resolution gammatone spectrograms (80–15 000 Hz, 128 channels) from stimulus `.wav` files.
- **`make_gt_predictor()`** — derives the full set of predictor variants (envelope, onset, 1-band, 8-band, linear, power-law) from a cached spectrogram.
- **`make_epochs()`** — legacy epoch extractor used for quick sanity checks.

### `plotting.py`
All visualisation functions, each accepting a `VARIABLES` configuration dict.

| Function | Description |
|----------|-------------|
| `get_yminmax` | Recursively find the global y-axis range across NDVars |
| `get_top_peaks` | Locate the N highest amplitude peaks in a mean TRF |
| `plot_trfs` | Butterfly + optional topomap panel for sustA / switA |
| `plot_topo` | Topographic map grid for attended vs. ignored TRFs |
| `plot_corr` | Grouped boxplot of backward-model correlations |
| `plot_masked_difference` | Significant attended-minus-ignored TRF difference |
| `plot_trfs_convA` | TRF butterfly plots specific to the convA paradigm |
| `plot_optLag` | Forward and backward optimal-lag analysis curves |

### `select_ica_gui.py`
Standalone script that launches the eelbrain ICA component selection GUI in a **separate process** to avoid conflicts with the Jupyter event loop. Called from `preprocessing.ipynb` via `subprocess.run()`. Temporarily copies the condition-specific ICA file to the generic location that eelbrain expects, opens the GUI, then copies the updated selections back.

---

## Running order

Run the three notebooks in the following order:

### 1. `preprocessing.ipynb`
- Computes gammatone spectrograms and predictor variables from stimulus audio.
- Converts raw XDF recordings to MNE FIF format.
- Fits ICA decompositions (scalp: 20 components, cEEGrid: 10 components).
- Opens the ICA selection GUI for manual artefact rejection.
- Extracts stimulus-locked epochs, resamples to 50 Hz, z-scores, and saves aligned EEG + predictor bundles to `data/derivatives/preprocessed/`.

### 2. `runTRFs.ipynb`
- Fits forward and backward TRF models for every subject, condition, sensor array, and predictor using `eelbrain.boosting` (L1 error, 50 ms basis, leave-one-trial-out cross-validation).
- Runs an optimal-lag sweep (45 ms window, 15 ms step, ±0.6 s range) to identify the peak integration lag for forward and backward models.
- Saves results to `results/`.

### 3. `analysis.ipynb`
- Loads the pre-fitted TRF bundles from `results/`.
- Generates all manuscript figures: TRF butterfly plots, topographic maps, masked attended-vs-ignored difference maps, backward-model correlation boxplots, convA conversation plots, and optimal-lag curves.
- Figures are saved as SVG and PNG under `figures/{condition}/`.

> **To reproduce the manuscript figures only**, skip steps 1–2 and run `analysis.ipynb` directly using the provided `results/*.pkl` files.

---

## Citation

If you use this code or dataset, please cite:

> *[Manuscript in preparation — citation will be added upon publication.]*

---

## License

See `LICENSE` for details.
