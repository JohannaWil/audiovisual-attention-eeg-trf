"""
experiment.py
=============
Defines the MobEEG TRFExperiment subclass and all preprocessing helpers used
to prepare EEG epochs and stimulus predictors for temporal response function
(TRF) modelling in the mobile-EEG auditory-attention study.

Main entry point
----------------
The module-level singleton ``mobEEG_e = MobEEG(DATA_ROOT)`` is imported by
the notebooks to run the full pipeline.  Direct usage::

    from experiment import mobEEG_e
    epochs, predictors, _, _, _ = mobEEG_e.align_epochs_and_predictors(...)

Key concepts
------------
* **Scalp** EEG: 44 channels recorded with a standard cap, average reference.
* **cEEGrid** EEG: 20 in-ear/around-ear channels, bipolar reference.
* **sustA / switA / convA**: three auditory-attention paradigm conditions
  (sustained, switching, conversational attention).
* **Predictors**: gammatone-derived temporal envelope and onset envelope for
  attended, ignored, and control (shifted) stimuli.
"""

from eelbrain.pipeline import *
from trftools.pipeline import *
import mne
import numpy as np
import pandas as pd
import json
import re
from pathlib import Path
import eelbrain as eel
import warnings


# ---------------------------------------------------------------------------
# Directory layout
# ---------------------------------------------------------------------------
ROOT = Path.cwd()                                             # project root (notebook working dir)
DATA_ROOT = Path.cwd() / 'data'                               # raw and derivative data
DATA_DER = DATA_ROOT / "derivatives"                          # all computed derivatives
DATA_DER.mkdir(parents=True, exist_ok=True)
LOG_DIR = DATA_DER / "logs"                                   # per-subject preprocessing logs
LOG_DIR.mkdir(parents=True, exist_ok=True)
PREDICTOR_DIR = ROOT / 'predictors'                           # gammatone predictor .pickle files
BEHAV_DIR = DATA_ROOT / "derivatives" / "behav"               # randomization CSVs

# Duration (seconds) of each recorded trial segment used for TRF fitting
SEGMENT_DURATION = {"all": 178}
# LSL trigger codes that mark trial onset for each condition
EVENT_CODES = {"sustA": 11, "switA": 21, "convA": 31}

# Default TRF model parameters shared across conditions
PARAMETERS = {
    "raw": "1-40_scalp",
    "samplingrate": 50,
    "data": "eeg",
    "tstart": -1.00,
    "tstop": 1.00,
    "filter_x": "continuous",
    "error": "l1",
    "basis": 0.050,
    "selective_stopping": 1,
}

# -------------------
# Logging utilities
# -------------------
def _log_append(
    log_rows: list,
    *,
    kind: str,
    subject: str,
    task: str,
    acquisition: str,
    item: str,
    step: str,
    **params,
):
    """Append one processing-step record to *log_rows*.

    Parameters
    ----------
    log_rows : list
        Accumulator list; rows are collected and written to CSV at the end.
    kind : str
        Data modality – one of ``'eeg'``, ``'predictor'``, or ``'mic'``.
    subject : str
        Subject identifier (e.g. ``'99'``).
    task : str
        Condition name – ``'sustA'``, ``'switA'``, or ``'convA'``.
    acquisition : str
        Sensor array – ``'scalp'`` or ``'ceegrid'``.
    item : str
        Human-readable label for the data object being processed.
    step : str
        Processing step label (e.g. ``'filter'``, ``'resample'``).
    **params
        Any additional key-value metadata to store alongside the record.
    """
    row = {
        "kind": kind,  # "eeg" | "predictor" | "mic"
        "subject": subject,
        "task": task,
        "acquisition": acquisition,
        "item": item,
        "step": step,
    }
    row.update(params)
    log_rows.append(row)

def _save_log_csv(log_rows: list, out_path: Path):
    """Persist *log_rows* to a CSV file, appending if the file already exists.

    Creates parent directories as needed so the caller does not have to.
    If *log_rows* is empty, the function returns immediately without touching
    the filesystem.
    """
    if not log_rows:
        return
    df = pd.DataFrame(log_rows)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # Append to existing CSV (keeps a rolling history across multiple runs)
    if out_path.exists():
        df.to_csv(out_path, mode="a", header=False, index=False)
    else:
        df.to_csv(out_path, index=False)

def _preprocess_pred_with_log(
    pred,
    *,
    low,
    high,
    target_fs,
    tmin,
    tmax,
    log_rows,
    subject,
    task,
    acquisition,
    item,
    regs=None,
    source="predictor",
    do_zscore: bool = True,
):
    """Band-pass filter, resample, optionally z-score, and time-crop a predictor.

    Each processing step appends a record to *log_rows* so the full provenance
    of every predictor is captured in the per-subject CSV log.

    Parameters
    ----------
    pred : eelbrain.NDVar
        Raw predictor loaded from disk (e.g. gammatone envelope).
    low, high : float
        Band-pass filter cut-offs in Hz.
    target_fs : int
        Target sampling rate in Hz (resampled from native rate).
    tmin, tmax : float
        Time window (seconds) to retain after resampling; aligned with the
        corresponding EEG epoch.
    log_rows : list
        Accumulator passed to ``_log_append``.
    subject, task, acquisition, item : str
        Provenance metadata written to the log.
    regs : str, optional
        Regressor name suffix (e.g. ``'~gammatone-1'``).
    source : str
        Data modality label stored in the log – ``'predictor'`` or ``'mic'``.
    do_zscore : bool
        If ``True`` (default), z-score the predictor after resampling.
    """
    _log_append(
        log_rows,
        kind=source,
        subject=subject,
        task=task,
        acquisition=acquisition,
        item=item,
        step="load",
        regs=regs,
    )

    pred = eel.filter_data(pred, low, high)
    _log_append(
        log_rows,
        kind=source,
        subject=subject,
        task=task,
        acquisition=acquisition,
        item=item,
        step="filter",
        regs=regs,
        l_freq=low,
        h_freq=high,
    )

    pred = eel.resample(pred, target_fs)
    _log_append(
        log_rows,
        kind=source,
        subject=subject,
        task=task,
        acquisition=acquisition,
        item=item,
        step="resample",
        regs=regs,
        target_fs=target_fs,
    )

    # Z-scoring is optional so mic predictors can skip it (Step B future work)
    if do_zscore:
        mu = float(pred.mean())
        sd = float(pred.std())
        pred = (pred - mu) / sd
        _log_append(
            log_rows,
            kind=source,
            subject=subject,
            task=task,
            acquisition=acquisition,
            item=item,
            step="zscore",
            regs=regs,
            mean=mu,
            std=sd,
        )
    else:
        _log_append(
            log_rows,
            kind=source,
            subject=subject,
            task=task,
            acquisition=acquisition,
            item=item,
            step="zscore_skip",
            regs=regs,
        )

    pred = pred.sub(time=(tmin, tmax))
    _log_append(
        log_rows,
        kind=source,
        subject=subject,
        task=task,
        acquisition=acquisition,
        item=item,
        step="crop",
        regs=regs,
        tmin=tmin,
        tmax=tmax,
    )

    return pred

# ---------------------------------------------------------------------------
# switA split helpers
# ---------------------------------------------------------------------------
# In the switA paradigm the listener switches attention between two talkers
# at fixed intervals.  The EEG epoch and every predictor must be sliced into
# the *same* three segments (before, during, and after the switch) so that
# the "attended" label is always aligned with the correct audio stream.

def _segments_to_samples(segments_sec, fs):
    """Convert a list of (start, end) time ranges in seconds to sample indices."""
    return [(int(a * fs), int(b * fs)) for a, b in segments_sec]

def _split_array_parts(data, seg_samples):
    """Slice a NumPy array along the last axis into named segment chunks."""
    part_names = [f"part{i+1}" for i in range(len(seg_samples))]
    return {p: data[..., s:e] for p, (s, e) in zip(part_names, seg_samples)}

def _split_eeg_parts_same(
    epochs_mne,
    *,
    segments_sec,
    target_fs,
    base_tmin,
    log_rows,
    subject,
    task,
    acquisition,
):
    """Slice an MNE EpochsArray into named time segments at the same boundaries.

    Used for the switA condition where the epoch must be split at attention-
    switch boundaries.  Each segment is stored as its own :class:`mne.EpochsArray`
    with a correctly offset ``tmin``.

    Parameters
    ----------
    epochs_mne : mne.EpochsArray
        Full-length resampled epoch array, shape ``(n_trials, n_channels, n_times)``.
    segments_sec : list[tuple[float, float]]
        List of ``(start, end)`` time ranges in seconds.
    target_fs : int
        Sampling rate in Hz used to convert seconds to sample indices.
    base_tmin : float
        Epoch start time (seconds) used to compute the per-part ``tmin`` offset.
    log_rows : list
        Provenance accumulator; one record is appended.
    subject, task, acquisition : str
        Metadata written to the log record.

    Returns
    -------
    epochs_parts : dict[str, mne.EpochsArray]
        Keys are ``'part1'``, ``'part2'``, etc., one entry per segment.
    """
    seg_samples = _segments_to_samples(segments_sec, target_fs)
    part_names = [f"part{i+1}" for i in range(len(seg_samples))]

    arr = epochs_mne.get_data()
    parts_arr = _split_array_parts(arr, seg_samples)

    epochs_parts = {}
    for p, (a, _b) in zip(part_names, segments_sec):
        epochs_parts[p] = mne.EpochsArray(parts_arr[p], epochs_mne.info, tmin=base_tmin + a)

    _log_append(
        log_rows,
        kind="eeg",
        subject=subject,
        task=task,
        acquisition=acquisition,
        item="epochs",
        step="crop_split_switA_same",
        segments=str(segments_sec),
        target_fs=target_fs,
        parts=",".join(part_names),
    )
    return epochs_parts

def _split_ndvar_parts_same(
    pred,
    *,
    segments_sec,
    target_fs,
    log_rows,
    subject,
    task,
    acquisition,
    item,
    regs,
    source_kind,
):
    """Slice an eelbrain NDVar into named time segments along the time dimension.

    Companion to :func:`_split_eeg_parts_same` for predictor NDVars.  Each
    output part carries the subset of the original time axis so that
    :func:`eelbrain.concatenate` can later re-assemble parts in the correct
    order.

    Parameters
    ----------
    pred : eelbrain.NDVar
        Predictor with ``time`` as the last dimension.
    segments_sec : list[tuple[float, float]]
        ``(start, end)`` time ranges in seconds defining the cut points.
    target_fs : int
        Sampling rate used to convert time ranges to sample indices.
    log_rows : list
        Provenance accumulator; one record is appended.
    subject, task, acquisition, item, regs : str
        Metadata written to the log record.
    source_kind : str
        Data-modality label stored in the log (e.g. ``'predictor'``).

    Returns
    -------
    out : dict[str, eelbrain.NDVar]
        Keys are ``'part1'``, ``'part2'``, etc., one entry per segment.

    Raises
    ------
    ValueError
        If the last dimension of *pred* is not ``'time'``.
    """
    seg_samples = _segments_to_samples(segments_sec, target_fs)
    part_names = [f"part{i+1}" for i in range(len(seg_samples))]

    if pred.dims[-1].name != "time":
        raise ValueError(f"Expected time as last dim, got dims={pred.dims}")

    x = pred.x
    out = {}
    for p, (s, e) in zip(part_names, seg_samples):
        x_part = x[..., s:e]
        time_part = pred.time[s:e]
        dims = list(pred.dims)
        dims[-1] = time_part
        out[p] = eel.NDVar(x_part, dims=dims, name=pred.name)

    _log_append(
        log_rows,
        kind=source_kind,
        subject=subject,
        task=task,
        acquisition=acquisition,
        item=item,
        step="crop_split_switA_same",
        segments=str(segments_sec),
        regs=regs,
        target_fs=target_fs,
    )
    return out

def _stage_task_specific_ica_for_eelbrain(data_root, subject, task, acq):
    """Copy the task-specific ICA solution to the generic filename Eelbrain expects.

    Eelbrain's pipeline looks for ICA files named
    ``sub-{subject}_acq-{acq}_eeg_raw-ica_ica.fif``, but we store one ICA
    solution per condition (task) to avoid cross-contamination.  This helper
    temporarily copies the correct file so the pipeline can find it without
    renaming the originals permanently.
    """
    ica_dir = Path(data_root) / "derivatives" / "ica"
    src = ica_dir / f"sub-{subject}_acq-{task}_{acq}_eeg_raw-ica_ica.fif"
    dst = ica_dir / f"sub-{subject}_acq-{acq}_eeg_raw-ica_ica.fif"
    if not src.exists():
        raise FileNotFoundError(f"ICA file not found:\n{src}")
    dst.parent.mkdir(parents=True, exist_ok=True)
    if not dst.exists() or src.stat().st_mtime > dst.stat().st_mtime:
        import shutil
        shutil.copy2(src, dst)
    return dst

# ---------------------------------------------------------------------------
# MobEEG TRF experiment definition
# ---------------------------------------------------------------------------

def _build_raw_pipelines(reject_criteria=None):
    """Build the dictionary of RawPipeline steps for all sensor arrays and tasks.

    Each entry in the returned dict is a named pipeline stage consumed by the
    Eelbrain TRFExperiment infrastructure.  The pipeline graph is:

        raw_{acq}  →  1-40_{acq}  →  {task}_ica_{acq}
                   →  0.1-40_{acq} →  0.1-40-ica_{task}_{acq}
                                   →  1-20_{task}_{acq}   (final band)

    Parameters
    ----------
    reject_criteria : dict or None
        Passed to ``RawICA`` for epoch-based artifact rejection during ICA
        fitting.  ``None`` disables rejection.
    """
    raw = {}

    acq_specs = {
        "scalp": {"n_components": 20, "final_band": (1, 20), "mid_band": (0.1, 40)},
        "ceegrid": {"n_components": 10, "final_band": (1, 20), "mid_band": (0.1, 40)},
    }

    tasks = ["sustA"] #, "switA", "convA"]

    for acq, spec in acq_specs.items():
        raw[f"raw_{acq}"] = RawSource(filename=f"{{subject}}_{{recording}}_{acq}-raw.fif")

        raw[f"1-40_{acq}"] = RawFilter(f"raw_{acq}", 1, 40)
        raw[f"0.1-40_{acq}"] = RawFilter(f"raw_{acq}", spec["mid_band"][0], spec["mid_band"][1])

        for task in tasks:
            ica_state = f"{task}_ica_{acq}"
            apply_state = f"0.1-40-ica_{task}_{acq}"

            raw[ica_state] = RawICA(
                f"1-40_{acq}",
                task,
                n_components=spec["n_components"],
                fit_kwargs={"reject": reject_criteria},
            )
            raw[apply_state] = RawApplyICA(f"0.1-40_{acq}", ica_state)

            low, high = spec["final_band"]
            raw[f"{low:g}-{high:g}_{task}_{acq}"] = RawFilter(apply_state, low, high)

    raw["1-20_scalp"] = raw.get("1-20_sustA_scalp")
    raw["1-20_ceegrid"] = raw.get("1-20_sustA_ceegrid")
    return raw


class MobEEG(TRFExperiment):
    """Mobile-EEG TRF experiment.

    Subclasses :class:`trftools.pipeline.TRFExperiment` and adds study-specific
    logic for:

    * loading behavioural randomization files to identify attended/ignored
      stimuli per trial;
    * splitting switA epochs and predictors at the attention-switch boundaries;
    * handling the conversational attention (convA) condition which uses
      spatially separated talkers instead of a dichotic paradigm.
    """

    data_dir = "eeg"          # subdirectory within DATA_ROOT that holds EEG files
    subject_re = r"sub\d\d"  # regex pattern identifying valid subject folders
    reject_criteria = None    # set a dict (e.g. {'eeg': 150e-6}) to enable rejection

    raw = _build_raw_pipelines(reject_criteria)

    def set(self, subject=None, match=True, allow_asterisk=False, **state):
        """Override ``TRFExperiment.set`` to allow combined acquisition names.

        Eelbrain's pipeline uses exact-match look-ups for the ``acquisition``
        field.  Passing ``match=False`` lets us set combined names such as
        ``'scalpSustA'`` that are not listed in the pipeline registry.
        """
        if 'acquisition' in state:
            match = False
        super().set(subject=subject, match=match, allow_asterisk=allow_asterisk, **state)

    def _randomization_path(self, subject: str, task: str) -> Path:
        """Return the path to the randomization CSV for *subject* and *task*.

        Each randomization file contains one row per trial with columns
        ``AttendedFile``, ``IgnoredFile``, ``TrialNumber``, and (for sustA)
        ``LRattention`` indicating which ear was cued.
        """
        subj_folder = f"sub-{subject}"
        if task == "sustA":
            return BEHAV_DIR / subj_folder / "randomization1.csv"
        if task == "switA":
            return BEHAV_DIR / subj_folder / "randomization2.csv"
        if task == "convA":
            return BEHAV_DIR / subj_folder / "randomization3.csv"
        raise ValueError(f"Unknown task: {task}")

    def get_predictor_names(self, subject: str, task: str):
        """Read the randomization file and return per-trial stimulus file lists.

        Returns
        -------
        predictor_names : dict
            Keys ``'attended_files'``, ``'ignored_files'``, ``'shifted_files'``
            each holding a list of .wav filenames (one per trial).
        left_trials : list[int] or None
            Trial indices where attention was cued to the left ear (sustA only).
        right_trials : list[int] or None
            Trial indices where attention was cued to the right ear (sustA only).
        attended_conv : list[bool] or None
            Per-trial flag indicating whether the attended talker is the front
            speaker (convA only).
        """
        df = pd.read_csv(self._randomization_path(subject, task))
        predictor_names = {
            "attended_files": df["AttendedFile"].to_list(),
            "ignored_files": df["IgnoredFile"].to_list(),
            "shifted_files": np.roll(df["IgnoredFile"].to_list(), shift=2),
        }

        left_trials = right_trials = None
        if task == "sustA":
            left_trials = df.loc[df["LRattention"] == "Left", "TrialNumber"].tolist()
            right_trials = df.loc[df["LRattention"] == "Right", "TrialNumber"].tolist()

        attended_conv = None
        if task == "convA":
            attended_conv = [str(x).startswith("trial") for x in predictor_names["attended_files"]]

        return predictor_names, left_trials, right_trials, attended_conv

    def align_epochs_and_predictors(
        self,
        typeOfRegressors,
        micRegressors,
        filters,
        condition,
        convSingle_attended=True,
        save_log=True,
    ):
        """Prepare time-aligned EEG epochs and stimulus predictors for TRF fitting.

        Loads the raw EEG for the currently selected subject/acquisition,
        extracts stimulus-locked epochs using the event file, resamples to
        ``PARAMETERS['samplingrate']``, z-scores each channel, then loads and
        preprocesses the corresponding predictor time-series.

        For the switA condition the 178 s epoch is split at the attention-switch
        boundaries and re-assembled so that the "attended" predictor tracks the
        correct audio stream throughout.

        Parameters
        ----------
        typeOfRegressors : list[str]
            Predictor suffixes to load (e.g. ``['~gammatone-1', '~gammatone-on-1']``).
        micRegressors : list[str] or None
            Mic-channel predictor suffixes; pass ``None`` to skip mic predictors.
        filters : tuple[float, float]
            (low, high) band-pass filter cut-offs in Hz applied to predictors.
        condition : str
            Task condition – ``'sustA'``, ``'switA'``, or ``'convA'``.
        convSingle_attended : bool
            convA-specific flag (currently unused placeholder).
        save_log : bool
            Whether to write the per-step provenance log to ``LOG_DIR``.

        Returns
        -------
        epochs_norm : mne.EpochsArray
            Z-scored EEG epochs, resampled to ``target_fs``.
        predictors : dict
            Nested dict ``{key: {regs: [NDVar]}}``.  Keys are
            ``'attended_files'``, ``'ignored_files'``, ``'shifted_files'``.
        predictors_mic : dict or None
            Mic predictors keyed by microphone and laterality; ``None`` if
            *micRegressors* is falsy or no mic files were found.
        predictors_attended : None
            Reserved for future use.
        attended_conv : list[bool] or None
            Passed through from ``get_predictor_names``.
        """
        task = condition
        acq_combined = self.get("acquisition")  # e.g. 'scalpSustA'
        acq = re.match(r'^(scalp|ceegrid)', acq_combined).group(1)  # e.g. 'scalp'
        subject = self.get("subject")

        log_rows = []

        final_band = "1-20"
        raw_state = f"{final_band}_{task}_{acq}"
        self.set(raw=raw_state)
        _log_append(log_rows, kind="eeg", subject=subject, task=task, acquisition=acq_combined, item="raw", step="select_raw_state", raw_state=raw_state)

        event_code = EVENT_CODES[task]
        low, high = filters

        tmin = 0
        target_fs = PARAMETERS["samplingrate"]
        desired_samples = SEGMENT_DURATION["all"] * target_fs
        tmax_sec = tmin + SEGMENT_DURATION["all"]
        crop_tmax = desired_samples / target_fs

        raw = self.load_raw()
        _log_append(log_rows, kind="eeg", subject=subject, task=task, acquisition=acq_combined, item="raw", step="load_raw")

        event_path = DATA_ROOT / f"sub-{subject}" / "eeg" / f"sub-{subject}_acq-{task}_events-eve_eeg.fif"

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning, message=".*does not conform to MNE naming conventions.*")
            event_array = mne.read_events(event_path)

        # Read event sample indices from the pre-saved MNE events file.
        # Column 0 contains sample indices in the EEG file's time base.
        # The previous XDF timestamp alignment was broken: event_times and the EEG
        # time axis use different clock references so all events mapped to sample 0.
        mask = event_array[:, 2] == event_code
        event_array_new = event_array[mask].copy()

        # Drop duplicate sample positions (keep first occurrence)
        _, keep_idx = np.unique(event_array_new[:, 0], return_index=True)
        event_array_new = event_array_new[np.sort(keep_idx)]

        print(f"Found {len(event_array_new)} events for subject {subject}, task {task}, code {event_code}")

        epochs = mne.Epochs(raw, event_array_new, tmin=tmin, tmax=tmax_sec, preload=True, baseline=(0, 0), verbose=False, event_repeated="drop")
        epochs.resample(target_fs)

        epochs_array = epochs.get_data()
        mean = epochs_array.mean(axis=2, keepdims=True)
        std = epochs_array.std(axis=2, keepdims=True)
        normalized_array = (epochs_array - mean) / std

        predictor_names, left_trials, right_trials, attended_conv = self.get_predictor_names(subject, task)
        if condition == 'switA':
            # switA switch boundaries (seconds): attention alternates between
            # talkers at t=35 s and t=125 s, with silent gaps at t=55 and t=145.
            # Segments [0-35], [55-125], [145-178] correspond to
            # attended=left, attended=right, attended=left respectively.
            segments = [[0,35],[55,125],[145,178]]
            segment_samples = [(int(start * target_fs), int(end * target_fs)) for start, end in segments]

            # Crop and concatenate each epoch along the time axis, removing
            # the transition/silence periods between attention switches
            cropped_epochs = []
            for epoch in normalized_array:
                parts = [epoch[:, start:end] for start, end in segment_samples]
                combined = np.concatenate(parts, axis=1)  # concat längs tidsaxeln
                cropped_epochs.append(combined)

         
            cropped_array = np.stack(cropped_epochs)  # form: (9, 44, new_time)
            epochs_norm = mne.EpochsArray(cropped_array, epochs.info, tmin=tmin)

            # predictor_names = self.get_predictor_names(config,subject)
            nTrials = len(predictor_names['attended_files'])
            predictors = {
                'attended_files': {},
                'ignored_files': {},
                'shifted_files': {},
            }
            
            for regs in typeOfRegressors:
                predictors['attended_files'][regs] = []
                predictors['ignored_files'][regs] = []
                predictors['shifted_files'][regs] = []

                true_attended = []
                true_ignored = []
                shifted = []
                for trial in range(nTrials):
                    
                    attended = predictor_names['attended_files'][trial]
                    name_att = attended[:-4]
                    dst = PREDICTOR_DIR / f'{name_att}{regs}.pickle'
                    att = eel.load.unpickle(dst)
                    att = eel.filter_data(att, low, high)
                    att = eel.resample(att,target_fs) 
                    mean = att.mean()
                    std = att.std()
                    att = (att - mean) / std
                    att1 = att.sub(time=(segments[0][0],segments[0][1])) 
                    att2 = att.sub(time=(segments[1][0],segments[1][1])) 
                    att3 = att.sub(time=(segments[2][0],segments[2][1])) 


                    ignored = predictor_names['ignored_files'][trial]
                    name_ign = ignored[:-4]
                    dst = PREDICTOR_DIR / f'{name_ign}{regs}.pickle'
                    ign = eel.load.unpickle(dst)
                    ign = eel.filter_data(ign, low, high)
                    ign = eel.resample(ign,target_fs) 
                    mean = ign.mean()
                    std = ign.std()
                    ign = (ign - mean) / std
                    ign1 = ign.sub(time=(segments[0][0],segments[0][1])) 
                    ign2 = ign.sub(time=(segments[1][0],segments[1][1])) 
                    ign3 = ign.sub(time=(segments[2][0],segments[2][1])) 

                    true_attended.append(eel.concatenate([att1,ign2,att3],dim='time'))
                    true_ignored.append(eel.concatenate([ign1,att2,ign3],dim='time'))

                    tmax = true_attended[0].x.shape[0]

                    fname = predictor_names['shifted_files'][trial]
                    name_shift = fname[:-4]
                    dst = PREDICTOR_DIR / f'{name_shift}{regs}.pickle'
                    shift = eel.load.unpickle(dst)
                    shift = eel.filter_data(shift, low, high)
                    shift = eel.resample(shift,target_fs) 
                    mean = shift.mean()
                    std = shift.std()
                    shift = (shift - mean) / std
                    shift = shift.sub(time=(0,tmax/target_fs))
                    shifted.append(shift)


                predictors['attended_files'][regs].append(eel.combine(true_attended))
                predictors['ignored_files'][regs].append(eel.combine(true_ignored))
                predictors['shifted_files'][regs].append(eel.combine(shifted))
        else:
            epochs_norm = mne.EpochsArray(normalized_array, epochs.info, tmin=tmin)

            

            predictors = {}

            for key, files in predictor_names.items():
                # if task != "switA":
                predictors[key] = {}

                for regs in typeOfRegressors:
                    pred_tmp = []

                    for fname in files:
                        name = fname[:-4]
                        if task == "convA" and name.startswith("trial"):
                            pred_path = PREDICTOR_DIR / f"{name}_mean{regs}.pickle"
                        else:
                            pred_path = PREDICTOR_DIR / f"{name}{regs}.pickle"

                        pred = eel.load.unpickle(pred_path)
                        pred = _preprocess_pred_with_log(
                            pred,
                            low=low,
                            high=high,
                            target_fs=target_fs,
                            tmin=tmin,
                            tmax=crop_tmax,
                            log_rows=log_rows,
                            subject=subject,
                            task=task,
                            acquisition=acq,
                            item=pred_path.name,
                            regs=regs,
                            source="predictor",
                            do_zscore=True,  # <-- change here later if you implement Step B
                        )

                        pred_tmp.append(pred)

                    predictors[key][regs] = [eel.combine(pred_tmp)]

        predictors_mic = None
        if micRegressors:
            mic_keys = ["Mic1", "Mic2", "Mic1_attLeft", "Mic1_attRight", "Mic2_attLeft", "Mic2_attRight"]

            predictors_mic = {k: {} for k in mic_keys}
            for k in predictors_mic:
                for regs in micRegressors:
                    predictors_mic[k][regs] = []

            if left_trials is not None and right_trials is not None:
                _log_append(
                    log_rows,
                    kind="mic",
                    subject=subject,
                    task=task,
                    acquisition=acq,
                    item="trials",
                    step="set_left_right_trials",
                    n_left=len(left_trials),
                    n_right=len(right_trials),
                )

            mics = []
            f1 = DATA_DER / f"sub-{subject}" / "eeg" / f"sub-{subject}_task-{task}_preds_mic1_eeg.pkl"
            f2 = DATA_DER / f"sub-{subject}" / "eeg" / f"sub-{subject}_task-{task}_preds_mic2_eeg.pkl"
            if f1.exists():
                mics.append(eel.load.unpickle(f1))
            if f2.exists():
                mics.append(eel.load.unpickle(f2))

            if mics:
                _log_append(log_rows, kind="mic", subject=subject, task=task, acquisition=acq, item=f1.name, step="unpickle", path=str(f1))
                _log_append(log_rows, kind="mic", subject=subject, task=task, acquisition=acq, item=f2.name, step="unpickle", path=str(f2))

                for m_idx, mic in enumerate(mics, start=1):
                    for regs in micRegressors:
                        pred = mic[regs]
                        pred = _preprocess_pred_with_log(
                            pred,
                            low=low,
                            high=high,
                            target_fs=target_fs,
                            tmin=tmin,
                            tmax=crop_tmax,
                            log_rows=log_rows,
                            subject=subject,
                            task=task,
                            acquisition=acq,
                            item=f"Mic{m_idx}:{regs}",
                            regs=regs,
                            source="mic",
                        )

                        key = f"Mic{m_idx}"
                        predictors_mic[key][regs].append(pred)

                        if left_trials is not None:
                            predictors_mic[f"Mic{m_idx}_attLeft"][regs].append(pred.sub(case=left_trials))
                            predictors_mic["trials_attLeft"] = left_trials
                            _log_append(log_rows, kind="mic", subject=subject, task=task, acquisition=acq, item=f"Mic{m_idx}:{regs}", step="subset_left", n_cases=len(left_trials))

                        if right_trials is not None:
                            predictors_mic[f"Mic{m_idx}_attRight"][regs].append(pred.sub(case=right_trials))
                            predictors_mic["trials_attRight"] = right_trials
                            _log_append(log_rows, kind="mic", subject=subject, task=task, acquisition=acq, item=f"Mic{m_idx}:{regs}", step="subset_right", n_cases=len(right_trials))


        predictors_attended = None
        if save_log:
            log_path1 = LOG_DIR / f"preproc_sub-{subject}_task-{task}_acq-{acq}.csv"
            _save_log_csv(log_rows, log_path1)

        return epochs_norm, predictors, predictors_mic, predictors_attended, attended_conv


mobEEG_e = MobEEG(DATA_ROOT)