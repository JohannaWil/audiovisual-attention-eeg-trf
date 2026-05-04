"""
select_ica_gui.py
=================
Launch the eelbrain ICA component selection GUI for a single subject.

Usage (from a notebook cell)::

    import subprocess
    subprocess.run([sys.executable, 'select_ica_gui.py', subject, condition, acq])

Arguments
---------
subject : str
    Subject identifier without the ``sub-`` prefix (e.g. ``'99'``).
condition : str
    Task condition – ``'sustA'``, ``'switA'``, or ``'convA'``.
acq : str
    Sensor array – ``'scalp'`` or ``'ceegrid'``.

Why a subprocess?
-----------------
eelbrain’s ``gui.run()`` starts a wx ``MainLoop`` which conflicts with the
IPython/Jupyter event loop already running inside the notebook kernel.  By
launching the GUI in a separate Python process the wx event loop runs cleanly
and the window behaves correctly.  The script blocks until the user closes the
window, at which point the selected ICA solution is copied back to the
task-specific file.

File management
---------------
Eelbrain looks for ICA files at the generic path
    ``ica/sub-{subject}_acq-{acq}_eeg_raw-ica_ica.fif``
but this study stores one solution per condition to prevent cross-contamination:
    ``ica/{condition}/sub-{subject}_acq-{acq}_eeg_raw-ica_ica.fif``
The script temporarily copies the condition-specific file to the generic
location before launching the GUI, then copies it back (with any new component
selections) before deleting the temporary generic file.
"""
import sys
import shutil
from pathlib import Path

# Parse command-line arguments passed by the calling notebook
subject, condition, acq = sys.argv[1], sys.argv[2], sys.argv[3]

from experiment import MobEEG, DATA_ROOT
from eelbrain import gui

# Paths for the condition-specific and generic ICA files
ica_dir = DATA_ROOT / 'derivatives' / 'ica'
cond_ica  = ica_dir / condition / f'sub-{subject}_acq-{acq}_eeg_raw-ica_ica.fif'
generic_ica = ica_dir / f'sub-{subject}_acq-{acq}_eeg_raw-ica_ica.fif'

e = MobEEG(DATA_ROOT)

try:
    # Stage the correct ICA file where Eelbrain can find it
    shutil.copy2(cond_ica, generic_ica)
    e.set(subject=subject, raw=f'{condition}_ica_{acq}', acquisition=acq)
    e.make_ica_selection()  # opens the GUI for the ICA fitted to this recording
    gui.run()  # blocks until the user closes the window
    # Persist any component selections back to the condition-specific file
    shutil.copy2(generic_ica, cond_ica)
    print(f'Saved: {condition}/{cond_ica.name}')
finally:
    # Always clean up the temporary generic file to avoid stale data
    if generic_ica.exists():
        generic_ica.unlink()
