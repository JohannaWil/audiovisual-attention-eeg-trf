"""
helpers.py
==========
Utility functions for the mobile-EEG auditory-attention study.

Covers:
* Loading and converting raw XDF recordings to MNE .fif format.
* Computing gammatone spectrograms and derived acoustic predictors.
* Epoch extraction and z-scoring helper used during offline TRF analysis.
* Data-structure factories for accumulating per-subject TRF results.

All file paths are resolved relative to the project root (``Path.cwd()``).
"""

import numpy as np
import pandas as pd
import random
import pathlib
from pathlib import Path
import pyxdf
import pickle
import mne
from matplotlib import pyplot as plt
from scipy.signal import butter,sosfilt,hilbert
from scipy.io.wavfile import read
import json
import eelbrain as eel

# config_path = 'config_mobEEG.json'
# with open(config_path) as f:
#     config = json.load(f)

ROOT = Path.cwd()                  # project root directory
DATA_ROOT = ROOT / 'data'          # raw + derivative data
STIMULUS_DIR = ROOT / 'stimuli'    # .wav audio files
PREDICTOR_DIR = ROOT / 'predictors'  # gammatone predictor .pickle files


def getint(cond):
    """Map a condition name string to an integer index (1-based).

    Used when a numeric condition code is needed for downstream processing.
    Returns 1 for ``'sustA'``, 2 for ``'switA'``, 3 for anything else.
    """
    cond_names = ["sustA","switA","convA"]
    if cond.lower() == cond_names[0].lower():
        return 1
    if cond.lower() == cond_names[1].lower():
        return 2
    else:
        return 3


def save_xdf_as_fif(condition,subject):
    """
    Loads EEG data, audio triggers and other LSL streams, as specified by the config file that should be placed in this folder.
    ----------
    Returns
    raw : mne raw Object 
        Raw object with correct montage in MNE.
    time : np.array
        Time array for the experiment
    events : np.array
        Trigger stream, values.
    event_times : np.array
        Trigger stream, time stamps.
    other_LSL : list with np.array elements
        List of any other LSL stream, values.
    other_LSL_times : list with np.array elements
        List of any other LSL stream, time stamps.
    other_LSL_descriptions : list with np.array elements
        List of any other LSL stream, descriptions.
    positions : not done.
    position_times : not done.
    """

    sub_name = f'sub-{subject}'
    
    EEG_DIR = DATA_ROOT / sub_name / 'eeg'
    DERIVATIVE_DIR = DATA_ROOT / 'derivatives'
    
    # save_path = Path(config["data_root"]) /  Path(subject) / Path("eeg")
    # derivative_path = Path(config["data_root"]) / Path("derivatives") / Path("mobEEG") / Path(subject) / Path("eeg")

    # montage_path = Path('E:\Data\ELLIIT\mobileEEG') / Path(config["montage_file"]) 
    data, header = pyxdf.load_xdf(EEG_DIR / f'{sub_name}_task-{condition}_eeg.xdf', synchronize_clocks=False)

    # Manually apply per-stream clock synchronization so all timestamps share
    # the same reference clock. pyxdf's synchronize_clocks=True is broken for this
    # file (bug: footer sample count stored as float string '1046071.0').
    # clock_times/clock_values hold the offset measurements (master - sender clock).
    for stream in data:
        ct = stream.get("clock_times", [])
        cv = stream.get("clock_values", [])
        if len(ct) > 0 and len(stream["time_stamps"]) > 0:
            corrections = np.interp(
                stream["time_stamps"], ct, cv,
                left=cv[0], right=cv[-1]
            )
            stream["time_stamps"] = stream["time_stamps"] + corrections

    print(f'Subject {subject}, condition {condition}, data streams found: {[d["info"]["name"][0] for d in data]}')

    dist_to_speaker = float(np.sqrt(0.84**2 + 1.17**2))
    # soundDelay = dist_to_speaker / 343
    soundDelay = 0#0.5
    print(f'Sound delay: {soundDelay}')

    scalp_reference = 'average'
    ceegrid_reference = ['L4', 'R4']


    eeg = []
    time = []
    events = []
    event_times = []
    tobii = []
    tobii_times = []
    tobii_description = []
    audio = []
    audio_times = []
    audio_description = []
    other_LSL = []
    other_LSL_times = []
    other_LSL_description = []
    fs = []
    for i in range(len(data)):
        if data[i]["info"]['name'][0] == 'EEG':
            eeg = data[i]["time_series"].T
            time = data[i]["time_stamps"]
            fs = int(float(data[i]["info"]['nominal_srate'][0]))
        elif data[i]["info"]['name'][0] == 'trigger stream':
            events = data[i]["time_series"][:,0]
            event_times = data[i]["time_stamps"]
        elif data[i]["info"]['name'][0] == 'Tobii':
            tobii.append(data[i]["time_series"])
            tobii_times.append(data[i]["time_stamps"])
            tobii_description.append(data[i]["info"])

        elif data[i]["info"]['name'][0] == 'MyAudioStream':
            audio.append(data[i]["time_series"])
            audio_times.append(data[i]["time_stamps"])
            audio_description.append(data[i]["info"])

        else:
            other_LSL.append(data[i]["time_series"])
            other_LSL_times.append(data[i]["time_stamps"])
            other_LSL_description.append(data[i]["info"])

    print("EEG shape:", eeg.shape)
    print(f"channel names: {data[i]['info']['desc'][0]['channels'][0]['channel']}")
    # Make montage! 
    montage_raw = mne.channels.read_custom_montage(DERIVATIVE_DIR / 'montage.sfp') 
    selected_ch_names = montage_raw.ch_names[:64]
    eeg = eeg[:64,:]

    ch_pos = {ch_name: pos for ch_name, pos in montage_raw.get_positions()['ch_pos'].items() if ch_name in selected_ch_names}
    montage = mne.channels.make_dig_montage(ch_pos=ch_pos, coord_frame='head')

    # Make the raw struct!
    info = mne.create_info(montage.ch_names, fs,ch_types="eeg")
    raw = mne.io.RawArray(eeg, info,verbose=False)
    raw.set_montage(montage)
    notchFreqs = np.arange(50, fs/2, 50)

    print("filtering")
    raw.notch_filter(notchFreqs)
    print(raw.info)

    scalp_channels = raw.info['ch_names'][:44]
    ceegrid_channels = raw.info['ch_names'][44:]

    # Scalp dataset
    raw_scalp = raw.copy().pick_channels(scalp_channels)
    raw_scalp.set_montage(montage) 
    raw_scalp.set_eeg_reference(ref_channels=scalp_reference)
    condition_cap = condition[0].upper() + condition[1:]
    scalp_file = EEG_DIR / f'{sub_name}_acq-scalp{condition_cap}_eeg.fif' 
    raw_scalp.save(scalp_file, overwrite=True)

    # cEEGrid dataset
    raw_ceegrid = raw.copy().pick_channels(ceegrid_channels)
    raw_ceegrid.set_montage(montage) 
    raw_ceegrid.set_eeg_reference(ref_channels=ceegrid_reference)
    ceegrid_file = EEG_DIR / f'{sub_name}_acq-ceegrid{condition_cap}_eeg.fif' 
    raw_ceegrid.save(ceegrid_file, overwrite=True)

    event_codes = events.astype(int)
    # Convert event timestamps to EEG sample indices.
    # Both streams are now in the same clock domain (synchronize_clocks=True).
    # searchsorted gives the index into the EEG time array, which matches
    # the RawArray saved starting at sample 0.
    event_samples = np.searchsorted(time, event_times).astype(int)
    event_samples = np.clip(event_samples, 0, len(time) - 1)
    event_array = np.column_stack((event_samples, np.zeros(len(events), dtype=int), event_codes))
    ds = {
        'time': time,
        'event_times': event_times
    }

    with open(EEG_DIR / f'{sub_name}_acq-{condition}_eeg_times_eeg.pickle', 'wb') as f:
        pickle.dump(ds, f)
    mne.write_events(EEG_DIR / f'{sub_name}_acq-{condition}_events-eve_eeg.fif', event_array,overwrite=True)

    if tobii:
        ds = {
            'time_series': tobii,
            'time_stamps': tobii_times,
            'info': tobii_description
        }
        with open(EEG_DIR / f'{sub_name}_acq-{condition}_tobii_data_eeg.pkl', 'wb') as f:
            pickle.dump(ds, f)


    if audio:
        ds = {
            'time_series': audio,
            'time_stamps': audio_times,
            'info': audio_description
        }
        with open(EEG_DIR / f'{sub_name}_acq-{condition}_audioStream_eeg.pkl', 'wb') as f:
            pickle.dump(ds, f)



def make_gammatone(wav, name):
    """Compute and cache gammatone spectrograms for a stimulus waveform.

    Spectrograms are saved as ``.pickle`` files under ``STIMULUS_DIR``.
    For stereo stimuli whose names start with ``'trial'`` (conversational
    attention), separate left-channel, right-channel, and mean spectrograms
    are written.  For all other stimuli a single mean spectrogram is written.

    Parameters
    ----------
    wav : eelbrain.NDVar
        Audio waveform NDVar, optionally with a ``'channel'`` dimension for
        stereo signals.
    name : str
        Base filename without extension; used to construct output paths.
    """


    if name.startswith('trial'):
            wav_left = wav.sub(channel=0)  
            wav_right = wav.sub(channel=1)
            wav_mean = wav.mean('channel')

            dst = STIMULUS_DIR / f'{name}_left-gammatone.pickle'
            if not dst.exists():
                gt = eel.gammatone_bank(wav_left, 80, 15000, 128, location='left', tstep=0.001)
                eel.save.pickle(gt, dst)

            dst = STIMULUS_DIR / f'{name}_right-gammatone.pickle'
            if not dst.exists():
                gt = eel.gammatone_bank(wav_right, 80, 15000, 128, location='left', tstep=0.001)
                eel.save.pickle(gt, dst)

            dst = STIMULUS_DIR / f'{name}_mean-gammatone.pickle'
            if not dst.exists():
                gt = eel.gammatone_bank(wav_mean, 80, 15000, 128, location='left', tstep=0.001)
                eel.save.pickle(gt, dst)
                dst = STIMULUS_DIR / f'{name}-gammatone.pickle'
                eel.save.pickle(gt, dst)

    elif wav.has_dim('channel') and len(wav.get_dim('channel')) == 2:
            wav_mean = wav.mean('channel')
            dst = STIMULUS_DIR / f'{name}-gammatone.pickle'
            gt = eel.gammatone_bank(wav_mean, 80, 15000, 128, location='left', tstep=0.001)
            eel.save.pickle(gt, dst)
    else:
            dst = STIMULUS_DIR / f'{name}-gammatone.pickle'
            gt = eel.gammatone_bank(wav, 80, 15000, 128, location='left', tstep=0.001) 
            eel.save.pickle(gt, dst)


def halfWaveRectifiedDerivative(xin,polyorder=3,win_len=21):
    """
    Half-wave rectified derivative of a signal.
    
    This function computes the half-wave rectified derivative of the input signal.
    It returns the positive part of the derivative, effectively removing negative values.
    """
    # Use Savitzky-Golay filter for smoothing and differentiation
    x = xin.get_data()
    from scipy.signal import savgol_filter
    y = savgol_filter(x, window_length=21, polyorder=polyorder, deriv=1, axis=1)
    x = xin.copy()
    x.data = y.clip(min=np.max(y)*0.01)
    return x

def make_gt_predictor(i, fname):
    """Derive and save all standard predictor variants from a gammatone spectrogram.

    Reads the high-resolution gammatone spectrogram ``{i}.pickle`` from
    ``STIMULUS_DIR`` and writes the following files to ``PREDICTOR_DIR``:

    * ``{fname}~gammatone-1.pickle``     – log-scale temporal envelope (1 band)
    * ``{fname}~gammatone-on-1.pickle``  – onset envelope (1 band, edge detector)
    * ``{fname}~gammatone-8.pickle``     – log-scale envelope (8 frequency bands)
    * ``{fname}~gammatone-on-8.pickle``  – onset envelope (8 frequency bands)
    * ``{fname}~gammatone-lin-8.pickle`` – linear-scale spectrogram (8 bands)
    * ``{fname}~gammatone-pow-8.pickle`` – power-law (x^0.6) spectrogram (8 bands)
    * ``{fname}~gammatone-onDer-1.pickle`` – half-wave rectified derivative (1 band)

    Parameters
    ----------
    i : str
        Stem of the cached gammatone ``.pickle`` file inside ``STIMULUS_DIR``.
    fname : str
        Prefix used for all output predictor filenames.
    """

    # Load the high resolution gammatone spectrogram
    gt = eel.load.unpickle(STIMULUS_DIR / f'{i}.pickle')

    # Apply a log transform to approximate peripheral auditory processing
    gt_log = (gt + 1).log()
    # Apply the edge detector model to generate an acoustic onset spectrogram
    gt_on = eel.edge_detector(gt_log, c=30)

    # Apply a temporal derivative version of the onset extraction // Oskar
    gt_on_der = halfWaveRectifiedDerivative(gt)
    eel.save.pickle(gt_on_der.sum('frequency'), PREDICTOR_DIR / f'{fname}~gammatone-onDer-1.pickle')

    # Create and save 1 band versions of the two predictors (i.e., temporal envelope predictors)
    eel.save.pickle(gt_log.sum('frequency'), PREDICTOR_DIR / f'{fname}~gammatone-1.pickle')

    eel.save.pickle(gt_on.sum('frequency'), PREDICTOR_DIR / f'{fname}~gammatone-on-1.pickle')
    # Create and save 8 band versions of the two predictors (binning the frequency axis into 8 bands)
    x = gt_log.bin(nbins=8, func='sum', dim='frequency')
    eel.save.pickle(x, PREDICTOR_DIR / f'{fname}~gammatone-8.pickle')

    x = gt_on.bin(nbins=8, func='sum', dim='frequency')
    eel.save.pickle(x, PREDICTOR_DIR / f'{fname}~gammatone-on-8.pickle')

    # Create gammatone spectrograms with linear scale, only 8 bin versions
    x = gt.bin(nbins=8, func='sum', dim='frequency')
    eel.save.pickle(x, PREDICTOR_DIR / f'{fname}~gammatone-lin-8.pickle')
    # Powerlaw scale
    gt_pow = gt ** 0.6
    x = gt_pow.bin(nbins=8, func='sum', dim='frequency')
    eel.save.pickle(x, PREDICTOR_DIR / f'{fname}~gammatone-pow-8.pickle')






def create_sustA_switA_structure(typeOfRegressors, sessions):
    """Create a nested dict for accumulating TRF results for sustA or switA.

    The returned structure has the shape::

        result[direction][sensor_array][attention][regressor] = []

    where ``direction`` is ``'fw'`` or ``'bw'``, ``sensor_array`` is
    ``'scalp'`` or ``'ceegrid'``, ``attention`` is one of the values in
    *sessions*, and ``regressor`` is one of the entries in *typeOfRegressors*.

    Parameters
    ----------
    typeOfRegressors : list[str]
        Predictor keys (e.g. ``['~gammatone-1', '~gammatone-on-1']``).
    sessions : list[str]
        Attention-condition labels (e.g. ``['attended', 'ignored', 'shifted']``).
    """
    structure = {}
    for direction in ['fw', 'bw']:
        structure[direction] = {}
        for location in ['scalp', 'ceegrid']:
            structure[direction][location] = {}
            for attention in sessions:
                structure[direction][location][attention] = {}
                for regs in typeOfRegressors:
                    structure[direction][location][attention][regs] = []
    return structure

def create_sustA_switA_structure2(typeOfRegressors):
    """Create a nested result dict for sustA / switA using the module-level ``sessions`` list.

    Identical in layout to :func:`create_sustA_switA_structure` but reads the
    attention-condition labels from the module-level ``sessions`` variable
    rather than requiring them to be passed as an argument.

    Parameters
    ----------
    typeOfRegressors : list[str]
        Predictor keys (e.g. ``['~gammatone-1', '~gammatone-on-1']``).

    Returns
    -------
    structure : dict
        Nested dict ``{direction: {sensor_array: {attention: {regressor: []}}}}``
        where the outer levels are ``'fw'`` / ``'bw'``, ``'scalp'`` /
        ``'ceegrid'``, and the attention labels come from ``sessions``.
    """
    structure = {}
    for direction in ['fw', 'bw']:
        structure[direction] = {}
        for location in ['scalp', 'ceegrid']:
            structure[direction][location] = {}
            for attention in sessions:
                structure[direction][location][attention] = {}
                for regs in typeOfRegressors:
                    structure[direction][location][attention][regs] = []
    return structure



def create_convA_structure(typeOfRegressors, sessions):
    """Create a nested dict for accumulating TRF results for the convA condition.

    Returns a dict with two top-level keys:

    * ``'convSingle'`` – for single-talker conversation trials, structured as
      ``result['convSingle'][direction][sensor][channel_type][regressor][attention]``.
    * ``'allTrials'`` – uses the same layout as ``create_sustA_switA_structure``.

    Parameters
    ----------
    typeOfRegressors : list[str]
        Predictor keys.
    sessions : list[str]
        Attention-condition labels.
    """
    structure = {}
    session = 'convSingle'
    structure[session] = {}
    for direction in ['fw', 'bw']:
        structure[session][direction] = {}
        for location in ['scalp', 'ceegrid']:
            structure[session][direction][location] = {}
            for chan in ['ConvMean', 'single']:
                structure[session][direction][location][chan] = {}
                for regs in typeOfRegressors:
                    structure[session][direction][location][chan][regs] = {}
                    for attention in ['attended', 'ignored']:
                        structure[session][direction][location][chan][regs][attention] = []
    session = 'allTrials'
    structure[session] = create_sustA_switA_structure(typeOfRegressors, sessions)

    return structure




def make_epochs(event_code, raw, events, event_times, time, tmax, ignore_nan=False, plot=False):
    """Extract fixed-length EEG epochs locked to events of a specific type.

    Finds the EEG sample index closest to each event timestamp, creates
    MNE Epochs, and returns a z-scored NumPy array.

    Parameters
    ----------
    event_code : int
        Trigger code to extract (e.g. ``11`` for sustA).
    raw : mne.io.Raw
        Continuous MNE Raw object.
    events : np.ndarray
        1-D array of integer trigger values from the LSL trigger stream.
    event_times : np.ndarray
        Timestamps (seconds) corresponding to each entry in *events*.
    time : np.ndarray
        EEG time-stamp array, same length as the EEG data.
    tmax : float
        Epoch duration in seconds (from event onset).
    ignore_nan : bool
        If ``True``, skip z-scoring (returns raw values instead).
    plot : bool
        If ``True``, display an interactive epochs browser.

    Returns
    -------
    epochs_array : np.ndarray, shape (n_epochs, n_times, n_channels)
        Z-scored (per-epoch, per-channel) EEG data.
    """
    xs = event_times[events==event_code]
    #print(xs)
    ys = time
    differences = np.abs(xs[:,np.newaxis] - ys[:,np.newaxis].T)
    minind = np.argmin(differences,axis=1)
    print("Trial starts found: ", len(minind))
    event_array = np.concatenate((minind[:,np.newaxis],np.zeros([len(xs),1]),events[events==event_code][:,np.newaxis]),axis=1)
    epochs = mne.Epochs(raw,event_array.astype(int),tmin=0,tmax=tmax,preload=True,baseline=(0,0),verbose=False)
    if plot:
        epochs.plot()
        plt.show()
    epochs_array = np.transpose(epochs.get_data(),[0,2,1])
    if ignore_nan:
        epochs_array = epochs_array
    if not ignore_nan:
        #print(epochs_array.shape)
        epochs_array = (epochs_array-np.mean(epochs_array,axis=1)[:,np.newaxis,:])/np.std(epochs_array,axis=1)[:,np.newaxis,:]
    return epochs_array