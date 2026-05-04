"""
plotting.py
===========
Visualization helpers for the mobile-EEG auditory-attention study.

All public functions accept a ``VARIABLES`` dict that bundles the plot
configuration (condition, sensor type, color scheme, save flags, etc.) so
callers do not need to forward a long list of keyword arguments.

Functions
---------
get_yminmax             recursively find the global y-axis range across NDVars.
get_top_peaks           locate the N highest amplitude peaks in a mean TRF.
plot_trfs               butterfly + optional topomap panel for sustA/switA.
plot_topo               topographic map grid for attended vs. ignored TRFs.
plot_corr               grouped boxplot of backward-model correlations.
plot_masked_difference  significant attended-minus-ignored TRF difference.
plot_trfs_convA         TRF butterfly plots specific to the convA paradigm.
plot_optLag             forward and backward optimal-lag analysis curves.
"""

import eelbrain as eel
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from scipy.signal import find_peaks
from matplotlib.patches import Patch
import seaborn as sns
from matplotlib.patches import Patch


# Human-readable display names used in figure titles and labels
NAMES = {
    '~gammatone-1': "Envelope",
    '~gammatone-on-1': "Onset",
    'sustA': "Sustained Attention",
    'switA': "Switching Attention",
    'convA': "Conversation Attention",
    'scalp': "Scalp sensors",
    'ceegrid': "cEEGrid sensors",
}

def get_yminmax(obj, global_min=float('inf'), global_max=float('-inf')):
    """Recursively determine the global y-axis limits across one or more NDVars.

    Traverses nested lists and dicts of :class:`eelbrain.NDVar` objects, opens
    a hidden Butterfly plot for each one to read its auto-scaled limits, and
    returns the union of all limits found.

    Parameters
    ----------
    obj : NDVar | list | dict
        A single NDVar or an arbitrarily nested collection of them.
    global_min, global_max : float
        Running extremes accumulated across recursive calls.

    Returns
    -------
    global_min, global_max : float
        Minimum and maximum y-values across all NDVars in *obj*.
    """
    if isinstance(obj, eel.NDVar):
        p = eel.plot.Butterfly(obj, show=False)
        y_min, y_max = p.get_ylim()
        p.close()
        global_min = min(global_min, y_min)
        global_max = max(global_max, y_max)

    elif isinstance(obj, list):
        for item in obj:
            global_min, global_max = get_yminmax(item, global_min, global_max)

    elif isinstance(obj, dict):
        for val in obj.values():
            global_min, global_max = get_yminmax(val, global_min, global_max)

    return global_min, global_max

def get_top_peaks(nd_mean, n_peaks=4, min_distance=2):
    """Return the time points of the *n_peaks* largest amplitude peaks.

    Computes the mean absolute amplitude across sensors, finds local maxima,
    and returns the *n_peaks* highest ones sorted by latency.

    Parameters
    ----------
    nd_mean : eelbrain.NDVar
        Mean TRF NDVar with ``sensor`` and ``time`` dimensions.
    n_peaks : int
        Number of peaks to return.
    min_distance : int
        Minimum sample distance between consecutive peaks.

    Returns
    -------
    peak_times : np.ndarray
        Times (seconds) of the selected peaks in descending amplitude order.
    """
    amplitude_over_time = nd_mean.abs('sensor').mean('sensor').x.squeeze()
    peaks, _ = find_peaks(amplitude_over_time, distance=min_distance)
    top_peaks = peaks[np.argsort(amplitude_over_time[peaks])[-n_peaks:][::-1]]
    return nd_mean.get_dim('time').times[top_peaks]


def plot_trfs(data_ndvar, regressor, VARIABLES):
    """Plot TRF butterfly traces for attended, ignored, and control conditions.

    Creates a two-panel figure (Attended | Ignored) showing all-channel TRF
    traces with optional highlighting of a channel subset.  When
    ``VARIABLES['bold_ave']`` is truthy, a second set of topomap panels is
    also produced.

    Parameters
    ----------
    data_ndvar : dict
        Nested dict ``{regressor: [attended_NDVar, ignored_NDVar, control_NDVar]}``.
    regressor : str
        Key into *data_ndvar* selecting the predictor to plot.
    VARIABLES : dict
        Plot configuration.  Required keys: ``sensor_type``, ``condition``,
        ``bold_ch``, ``bold_ave``, ``colors``, ``save_figs``, ``fig_dir``,
        ``cmap``.
    """

    sensors = VARIABLES['sensor_type']
    condition = VARIABLES['condition']
    bold_ch = VARIABLES['bold_ch']
    boldAve = VARIABLES['bold_ave']
    colors = VARIABLES['colors']
    save_figs = VARIABLES['save_figs']
    fig_dir = Path(VARIABLES['fig_dir'])
    cmap = VARIABLES['cmap']
    
    for bold in boldAve:

        fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
        global_min, global_max = float('inf'), float('-inf')

        # Add Control (gray, index 2)
        nd = data_ndvar[regressor][2].sub(time=(-0.3, 0.8))
        color = colors[2]  # Gray
        for ax in axes:
            nd_plot = nd
        
            nd_mean = nd_plot.mean('case')

            smoothed = nd_mean.smooth(dim='time', window_size=0.12, window='hamming')
            time = nd.get_dim('time')
            ax.plot(time, smoothed.x.T, color=color, alpha=0.4, linewidth=1)
            ax.plot([], [], color=color, label='Control', linewidth=4)

        nd_means = []
        ave_means = []
        for label, idx, color in zip(['Attended', 'Ignored'], [0, 1], colors[:2]):
            nd = data_ndvar[regressor][idx].sub(time=(-0.3, 0.8))
            nd_ave = data_ndvar[regressor][idx].sub(time=(-0.3, 0.8))

            time = nd.get_dim('time')
            nd_mean = nd.mean('case')
            nd_means.append(nd_mean)
            smoothed = nd_mean.smooth(dim='time', window_size=0.12, window='hamming')
            
            
            axes[idx].plot(time, smoothed.x.T, alpha=0.3, linewidth=1, color=color)
            global_min = min(global_min, smoothed.x.min())
            global_max = max(global_max, smoothed.x.max())

            

            if bold:
                nBoldCh = len(bold_ch)
                nd_ave = nd_ave.sub(sensor=bold_ch)
                nd_mean = nd_ave.mean('case')
                ave_means.append(nd_mean)
                smoothed = nd_mean.smooth(dim='time', window_size=0.12, window='hamming')
                if len(bold_ch) < 4:
                    avg = smoothed.x.T.squeeze()
                else:
                    avg = smoothed.mean('sensor').x.T.squeeze()
                
                axes[idx].plot(time, avg, alpha=1, linewidth=3, color=color)
                axes[idx].plot([], [], color=color, label=f'All 44 sensors', linewidth=4, alpha=0.3)
                axes[idx].plot([], [], color=color, label=f'Average {nBoldCh} sensors', linewidth=4)
            else:
                axes[idx].plot([], [], color=color, label=f'All 44 sensors', linewidth=4, alpha=0.3)

            axes[idx].set_title(f"{label}")
            axes[idx].set_xlabel("Time [s]")
            if idx == 0:
                axes[idx].set_ylabel("Amplitude [a.u.]")
            axes[idx].spines['top'].set_visible(False)
            axes[idx].spines['right'].set_visible(False)



        for ax in axes:
            ax.set_ylim(global_min, global_max)
            ax.legend()

        fig.suptitle(f"{NAMES[condition]} - {NAMES[sensors]} - {NAMES[regressor]}", fontsize=16)
        plt.tight_layout()
        plt.show()

        if save_figs:
            suffix = "boldAve" if bold else "noBoldAve"
            figname = f"{condition}_{sensors}_{NAMES[regressor]}_{suffix}"
            fig.savefig(fig_dir / f"{figname}.svg", format="svg")
            fig.savefig(fig_dir / f"{figname}.png", format="png")


        if bold:
            trf_dir = 'fw'
            titles = ['Attended', 'Ignored']

            global_min, global_max = float('inf'), float('-inf')

            for i in range(2):
                nd = data_ndvar[regressor][i].sub(time=(-0.3, 0.8))
                nd_mean = nd.mean('case')
                amp = nd_mean.abs('sensor').mean('sensor').x.squeeze()
                global_min = min(global_min, amp.min())
                global_max = max(global_max, amp.max())

            absmax = max(abs(global_min), abs(global_max))

            absmax = max(abs(global_min), abs(global_max))
            for i, nd_mean in enumerate(nd_means):
                if condition == 'sustA':
                    if regressor == '~gammatone-1':
                        if i == 0:
                            peak_times = [0.04, 0.1, 0.18,0.3]
                        else:
                            peak_times = [0.04, 0.2, 0.34]
                    else:
                        if i == 0:
                            peak_times = [0.06, 0.14, 0.24, 0.34]
                        else:
                            peak_times = [0.06, 0.38]
         
                if condition == 'switA':
                    if regressor == '~gammatone-1':
                        if i == 0:
                            peak_times = [0.06, 0.12, 0.18,0.3]
                        else:
                            peak_times = [0.04, 0.14, 0.38]
                    else:
                        if i == 0:
                            peak_times = [0.08, 0.14, 0.22, 0.32]
                        else:
                            peak_times = [0.08, 0.26, 0.44]
                elif condition == 'convA':
                    if regressor == '~gammatone-1':
                        if i == 0:
                            peak_times = [0.06, 0.12, 0.2,0.34]
                        else:
                            peak_times = [0.06, 0.22, 0.38]
                    else:
                        if i == 0:
                            peak_times = [0.08, 0.14, 0.23, 0.34]
                        else:
                            peak_times = [0.08, 0.26, 0.44]

                p = eel.plot.TopoArray(nd_mean, t=peak_times, cmap=cmap, vmin=-absmax, vmax=absmax)
                figtitle = f"{NAMES[condition]} - {trf_dir} - {NAMES[sensors]} - {NAMES[regressor]}: {titles[i]}"
                p.figure.suptitle(figtitle)
                p.plot_colorbar(label_rotation=270)
                
                if save_figs:
                    figname = f"{condition}_{trf_dir}_{sensors}_{NAMES[regressor]}_{titles[i]}_topos"
                    p.figure.savefig(fig_dir / f"{figname}.svg", format="svg")
                    p.figure.savefig(fig_dir / f"{figname}.png", format="png")
                print('---')


def plot_topo(data_ndvar, regressor, VARIABLES):
    """Plot topographic maps at automatically detected peak latencies.

    For each of the attended and ignored TRF means, detects the top amplitude
    peaks via :func:`get_top_peaks` and renders them as
    :class:`eelbrain.plot.TopoArray` panels.

    Parameters
    ----------
    data_ndvar : dict
        Nested dict ``{regressor: [attended_NDVar, ignored_NDVar, ...]}``.
    regressor : str
        Key into *data_ndvar*.
    VARIABLES : dict
        Required keys: ``sensor_type``, ``condition``, ``save_figs``,
        ``fig_dir``, ``cmap``, ``trf_dir``.
    """

    sensors = VARIABLES['sensor_type']
    condition = VARIABLES['condition']
    save_figs = VARIABLES['save_figs']
    fig_dir = Path(VARIABLES['fig_dir'])
    cmap = VARIABLES['cmap']
    trf_dir = VARIABLES['trf_dir']

    titles = ['Attended', 'Ignored']
    global_min, global_max = float('inf'), float('-inf')

    nd_means = []
    for i in range(2):
        nd = data_ndvar[regressor][i].sub(time=(-0.3, 0.8))
        nd_mean = nd.mean('case')
        nd_means.append(nd_mean)
        amp = nd_mean.abs('sensor').mean('sensor').x.squeeze()
        global_min = min(global_min, amp.min())
        global_max = max(global_max, amp.max())

    absmax = max(abs(global_min), abs(global_max))

    for i, nd_mean in enumerate(nd_means):
        peak_times = get_top_peaks(nd_mean)
        p = eel.plot.TopoArray(nd_mean, t=peak_times.tolist(), cmap=cmap, vmin=-absmax, vmax=absmax)
        figtitle = f"{NAMES[condition]} - {trf_dir} - {NAMES[sensors]} - {NAMES[regressor]}: {titles[i]}"
        p.figure.suptitle(figtitle)
        p.plot_colorbar(label_rotation=270)

        if save_figs:
            figname = f"{condition}_{trf_dir}_{sensors}_{NAMES[regressor]}_topos_{titles[i]}"
            p.figure.savefig(fig_dir / f"{figname}.svg", format="svg")
            p.figure.savefig(fig_dir / f"{figname}.png", format="png")


def plot_corr(data, VARIABLES):
    """Grouped boxplot comparing backward-model correlations across conditions.

    Shows one group of three boxes (Target / Masker / Control) for each of the
    four sensor × predictor combinations: scalp-envelope, scalp-onset,
    cEEGrid-envelope, cEEGrid-onset.

    Parameters
    ----------
    data : dict
        Nested dict ``{sensor: {attention: {regressor: [values_array]}}}``,
        where ``attention`` is one of ``'attended'``, ``'ignored'``,
        ``'shifted'``.
    VARIABLES : dict
        Required keys: ``condition``, ``colors``, ``save_figs``, ``fig_dir``.
    """

    condition = VARIABLES['condition']
    colors = VARIABLES['colors']
    save_figs = VARIABLES['save_figs']
    fig_dir = Path(VARIABLES['fig_dir'])


    condition_labels = ["Target", "Masker", "Control"]

    xtick_labels = [
        ("Scalp", "Envelope"),
        ("Scalp", "Onset"),
        ("cEEGrid", "Envelope"),
        ("cEEGrid", "Onset")
    ]

    group_names = [
        ("scalp", "~gammatone-1"),
        ("scalp", "~gammatone-on-1"),
        ("ceegrid", "~gammatone-1"),
        ("ceegrid", "~gammatone-on-1")
    ]

    box_data = []
    x_labels = []
    group_ticks = []
    group_positions = []

    position = 0
    width = 0.6
    spacing = 1.0

    for group_index, (site, feature) in enumerate(group_names):
        for i, cond in enumerate(["attended", "ignored", "shifted"]):

            values = data[site][cond][feature][0]  # extrahera vektorn
            box_data.append(values)
            x_labels.append(f"{site} {feature}")
            position_offset = position + i * width
            group_positions.append(position_offset)

        group_ticks.append(position + width)  # mitten för gruppens tre boxar
        position += 3 * width + spacing


    plt.figure(figsize=(12, 6))
    bp = plt.boxplot(box_data, positions=group_positions, patch_artist=True)

    for patch, color in zip(bp['boxes'], colors * 4):
        patch.set_facecolor(color)


    for median in bp['medians']:
        median.set_color("black")
        median.set_linewidth(2)

   
    plt.xticks(group_ticks, [f"{site}\n{feat}" for site, feat in xtick_labels],rotation=0,fontsize=16)
    plt.ylabel("Correlation", fontsize=14)
    plt.title(f"Correlations for backward models: {NAMES[condition]}", fontsize=18)

    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    ax.tick_params(axis='y', labelsize=14)


    # Legend
    legend_elements = [Patch(facecolor=colors[i], label=label) for i, label in enumerate(condition_labels)]
    plt.legend(handles=legend_elements,frameon=False,fontsize=16)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             

    # Layout och save
    plt.tight_layout()
    if save_figs:
        plt.savefig(fig_dir / f"{condition}_bw_corr.svg", format="svg")
        plt.savefig(fig_dir / f"{condition}_bw_corr.png", format="png")

    plt.show()


def plot_masked_difference(data_ndvar, regressor, VARIABLES):
    """Plot the attended-minus-ignored TRF difference masked by significance.

    Runs a TFCE-corrected independent-samples t-test between attended and
    ignored TRFs and plots only the time-sensor locations where the difference
    is significant at the level given by ``VARIABLES['alpha']``.

    Parameters
    ----------
    data_ndvar : dict
        Nested dict ``{regressor: [attended_NDVar, ignored_NDVar, ...]}``.
    regressor : str
        Key into *data_ndvar*.
    VARIABLES : dict
        Required keys: ``sensor_type``, ``condition``, ``save_figs``,
        ``fig_dir``, ``colors``, ``trf_dir``, ``alpha``.
    """

    sensors = VARIABLES['sensor_type']
    condition = VARIABLES['condition']
    save_figs = VARIABLES['save_figs']
    fig_dir = Path(VARIABLES['fig_dir'])
    colors = VARIABLES['colors']
    trf_dir = VARIABLES['trf_dir']
    alpha = VARIABLES['alpha'] 



    win_size = 0.1
    att = data_ndvar[regressor][0].sub(time=(-0.2, 0.8)).smooth(dim='time', window_size=win_size, window='hamming')
    ign = data_ndvar[regressor][1].sub(time=(-0.2, 0.8)).smooth(dim='time', window_size=win_size, window='hamming')
    result = eel.testnd.TTestIndependent(att, ign, tfce=True)

    p = eel.plot.Butterfly(result.masked_difference(p=alpha), color=colors[0], linewidth=1.5,vmin=-0.005,vmax=0.005)
    figtitle = f"{NAMES[condition]} - {NAMES[sensors]} - {NAMES[regressor]}  p={alpha}"
    p.figure.suptitle(figtitle)
    p.figure.axes[0].spines['top'].set_visible(False)
    p.figure.axes[0].spines['right'].set_visible(False)
    p.add_vline(0, color='gray', linestyle='--')


    if save_figs:
        figname = f"{condition}_{trf_dir}_{sensors}_{regressor}_maskedDiff"
        p.figure.savefig(fig_dir / f"{figname}.svg", format="svg")
        p.figure.savefig(fig_dir / f"{figname}.png", format="png")


def plot_trfs_convA(ndvars, regressor, VARIABLES):
    """Plot TRF butterfly traces for the conversational attention (convA) condition.

    Produces a 2×2 panel figure (front/side talker × attended/ignored) with
    individual-channel traces shown faintly and highlighted channels overlaid
    in bold.  Time axis is in milliseconds.

    Parameters
    ----------
    ndvars : dict
        Nested dict ``{convSingle: {'attended': [NDVar], 'ignored': [NDVar]}}``,
        where *convSingle* iterates over the two conversation configurations.
    regressor : str
        Regressor label used in the figure title.
    VARIABLES : dict
        Required keys: ``sensor_type``, ``condition``, ``save_figs``,
        ``conv_channels``, ``fig_dir``, ``ymin``, ``ymax``, ``convSession``.
    """
    
    sensors = VARIABLES['sensor_type']
    condition = VARIABLES['condition']
    save_figs = VARIABLES['save_figs']
    channels = VARIABLES['conv_channels']
    fig_dir = Path(VARIABLES['fig_dir'])
    ymin = VARIABLES['ymin']
    ymax = VARIABLES['ymax']
    session = VARIABLES['convSession']

    
    labels = ["Target", "Masker"]
    for ch in channels:
        labels.append(ch)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharey=True)
    axes = axes.ravel()

    colors = sns.color_palette("deep")
    picked_cc = [0,1,2,3,8]

    idxes = [0,2,1,3]
    i = 0
    for convSingle in ndvars.keys():
        ii = idxes[i]
        ax = axes[ii]
        nds = ndvars[convSingle]

        # --- attended ---
        nd = nds['attended'][0]
        nd = nd.sub(time=(-0.1, 0.6))
        nd_mean = nd.mean('case')
        
        smoothed = nd_mean.smooth(dim='time', window_size=0.12, window='hamming')

        time = nd.get_dim('time')
        time_ms = time.times * 1000
        smoothed.x = smoothed.x * 1000
        ax.plot(time_ms, smoothed.x.T, alpha=0.3, linewidth=1, color=colors[0])

        for ch_idx, ch in enumerate(channels):
            smoothed_ch = smoothed.sub(sensor=ch)
            c =  picked_cc[ch_idx+2]
            ax.plot(time_ms, smoothed_ch.x.T, alpha=1, linewidth=3, color=colors[c])
          
        i += 1
        ii = idxes[i]
        ax = axes[ii]
        # --- ignored ---
        nd = nds['ignored'][0]
        nd = nd.sub(time=(-0.1, 0.6))
        nd_mean = nd.mean('case')
        smoothed = nd_mean.smooth(dim='time', window_size=0.12, window='hamming')

        time = nd.get_dim('time')
        time_ms = time.times * 1000
        smoothed.x = smoothed.x * 1000
        ax.plot(time_ms, smoothed.x.T, alpha=0.3, linewidth=1, color=colors[1])

        for ch_idx, ch in enumerate(channels):
            smoothed_ch = smoothed.sub(sensor=ch)
            c =  picked_cc[ch_idx+2]
            ax.plot(time_ms, smoothed_ch.x.T, alpha=1, linewidth=3, color=colors[c])
        i += 1

    axes[0].set_title('Front conversation', fontsize=14)
    axes[1].set_title('Side talker', fontsize=14)

    for r in range(4):
        axes[r].axvline(0, linestyle='--', linewidth=0.5, color='k')
        axes[r].grid(True, alpha=0.2)
        axes[r].set_ylim(ymin*1000,ymax*1000)
  
        axes[r].spines['top'].set_visible(False)
        axes[r].spines['right'].set_visible(False)

        # Increase tick label font size
        axes[r].tick_params(axis='both', which='major', labelsize=14)

        # Remove x-ticks from top two figures
        if r in [0, 1]:
            axes[r].tick_params(axis='x', labelbottom=False)

        if r == 0 or r == 2:
            axes[r].set_ylabel('Amplitude [a.u.]', fontsize=20)
        if r == 2 or r == 3:
            axes[r].set_xlabel('Time [ms]', fontsize=20)


    
    plt.rcParams.update({'axes.labelsize': 18, 'xtick.labelsize': 18, 'ytick.labelsize': 18})
    legend_elements = [Patch(facecolor=colors[picked_cc[i]], label=label) for i, label in enumerate(labels)]
    plt.legend(handles=legend_elements,frameon=False,fontsize=16)  


    figtitle = f'{NAMES[condition]} - {NAMES[sensors]} - {NAMES[regressor]}'
    fig.suptitle(figtitle, fontsize=16)

    plt.tight_layout()
    plt.show()

    if save_figs:
        figname = f"{condition}_{session}_{sensors}_{regressor}"
        fig.savefig(fig_dir / f"{figname}.svg", format="svg")
        fig.savefig(fig_dir / f"{figname}.png", format="png")


def plot_optLag(optLag, reg, VARIABLES):
    """Plot optimal-lag analysis curves for forward and backward TRF models.

    Shows mean correlation as a function of time lag for each sensor array
    (scalp / cEEGrid) and attention condition (attended / ignored), with
    95 % confidence intervals shaded.  The backward model is plotted on the
    left axis and the forward model on the right.

    Parameters
    ----------
    optLag : dict
        Nested dict ``{direction: {sensor: {attention: {regressor: [NDVar]}}}}``,
        where *direction* is ``'fw'`` or ``'bw'``.
    reg : str
        Regressor key to plot (e.g. ``'~gammatone-1'``).
    VARIABLES : dict
        Required keys: ``condition``, ``save_figs``, ``fig_dir``, ``colors``.
    """

    condition = VARIABLES['condition']
    save_figs = VARIABLES['save_figs']
    fig_dir = Path(VARIABLES['fig_dir'])
    colors = VARIABLES['colors']
    sensors = ['scalp','ceegrid']
    attentions = ['attended','ignored']
    directions = ['fw','bw']

    if reg == '~gammatone-1':
        ymin, ymax = -0.01, 0.052
    else:
        ymin, ymax = -0.005, 0.035

    plotDesigns = {}
    linestyles = ['-','--']
    labels = ['Target scalp','Target ceegrid','Masker scalp','Masker ceegrid']

    i = 0
    for aa, attention in enumerate(['attended', 'ignored']):
        plotDesigns[attention] = {}
        for ll, location in enumerate(['scalp', 'ceegrid']):
            plotDesigns[attention][location] = {}
            plotDesigns[attention][location]['colors'] = colors[aa]
            plotDesigns[attention][location]['linestyles'] = linestyles[ll]
            plotDesigns[attention][location]['labels'] = labels[i]
            i += 1


    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
    
    for attention in attentions:
        for direction in directions:
            for sensor in sensors:

                data = optLag[direction][sensor][attention][reg]
              
                color = plotDesigns[attention][sensor]['colors']
                linestyle = plotDesigns[attention][sensor]['linestyles']
                label = plotDesigns[attention][sensor]['labels']
                time = data[0].get_dim('time')
                
                nSubs = data[0].x.shape[0]
                if direction == 'fw':
                    data_mean = data[0].mean(axis=('case','sensor'))
                    data_var = 1.96 * data[0].std(axis=('case','sensor')) / np.sqrt(nSubs) 
                
                elif direction == 'bw':
    
                    data_mean = data[0].mean(axis='case')
                    data_var = 1.96 * data[0].std(axis='case') / np.sqrt(nSubs) 
                    
            

                tmin = data[0].time.tmin
                tmax = data[0].time.tmax
                if direction == 'fw':

                    data_mean = data_mean.sub(time=(-0.03, tmax))
                    data_var = data_var.sub(time=(-0.03, tmax))
                    
                elif direction == 'bw':

                    data_mean = data_mean.sub(time=(tmin, 0.03))
                    data_var = data_var.sub(time=(tmin, 0.03))
                    

                data_mean = data_mean.smooth(dim='time', window_size=0.08, window='hamming')
                data_var = data_var.smooth(dim='time', window_size=0.08, window='hamming')

                tmin = -0.03 * 1000
                tstep = time.tstep * 1000
                nsamples = data_mean.x.shape[0]
                unit = 'ms'
                t = eel.UTS(tmin,tstep,nsamples,unit)

                mean_vals = data_mean.x  

                if direction == 'fw':
                    
                    axes[1].plot(t, mean_vals, color=color, alpha=1, linewidth=3,
                            label=label, linestyle=linestyle)
                    
                    axes[1].fill_between(t, mean_vals - data_var, mean_vals + data_var,
                                color=color, alpha=0.1)
                    
                    axes[1].set_ylim(ymin,ymax)
                    
                else:

                    axes[0].plot(t, mean_vals, color=color, alpha=1, linewidth=3,
                            linestyle=linestyle)
                    
                    axes[0].fill_between(t, mean_vals - data_var, mean_vals + data_var,
                                color=color, alpha=0.1)
                    
                    axes[0].set_ylim(ymin,ymax)
                    
        

    plt.legend(fontsize=14)

    for i,ax in enumerate(axes):
        ax.set_xlim(t[0]-25, t[-1]+25)
        ax.set_ylim(ymin, ymax)
    
        ax.set_xlabel("Time [ms]",fontsize=18)
        ax.tick_params(axis="x", labelsize=14)
        ax.tick_params(axis="y", labelsize=14)
        ax.axvline(0, color='gray', linestyle='--')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        if i == 1:
            ax.spines['left'].set_visible(False)
        ax.grid(True, linestyle='-', alpha=0.3)


    axes[0].set_ylabel("Correlation",fontsize=18)
    axes[0].set_title('Backward model', fontsize=14)
    axes[1].set_title('Forward model', fontsize=14)
    figtitle = f'Optimal lag analysis: {NAMES[condition]} - {NAMES[reg]}'
    fig.suptitle(figtitle, fontsize=16)

    plt.tight_layout()
    ax = plt.gca()
    
    # plt.title(f"{condition}", fontsize=18)

    if save_figs:
        figname = f"{condition}_{reg}_corr"
        plt.savefig(fig_dir / f"{figname}.svg", format="svg")
        plt.savefig(fig_dir / f"{figname}.png", format="png")

    plt.show()








