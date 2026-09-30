"""
utils.py
========
Shared signal processing utilities for hemodynamic correction and
behavioral prediction LSTM notebooks.

Functions are organized into four groups:
  1. Timestamp alignment
  2. Signal preprocessing
  3. Hemodynamic correction
  4. Analysis helpers (PSTH, statistics)
"""

import numpy as np
import pandas as pd
from scipy import interpolate, signal, stats
from scipy.signal import medfilt, convolve
from scipy.signal.windows import hamming


# ---------------------------------------------------------------------------
# 1. Timestamp alignment
# ---------------------------------------------------------------------------

def interpolate_to_new_timestamps(original_times, original_signal, new_times):
    """Resample a signal onto a new time grid via linear interpolation.

    Parameters
    ----------
    original_times : array-like
        Timestamps corresponding to original_signal.
    original_signal : array-like
        Signal values to resample.
    new_times : array-like
        Target timestamps.

    Returns
    -------
    np.ndarray
        Signal resampled at new_times.
    """
    original_times = np.asarray(original_times)
    original_signal = np.asarray(original_signal)

    # Trim to the shorter of the two if lengths differ slightly
    n = min(len(original_times), len(original_signal))
    original_times = original_times[:n]
    original_signal = original_signal[:n]

    f = interpolate.interp1d(
        original_times, original_signal,
        bounds_error=False, fill_value="extrapolate"
    )
    return f(new_times)


def interpolate_nans(trace):
    """Replace NaN values in a trace with linearly interpolated values.

    Parameters
    ----------
    trace : np.ndarray
        Input array (modified in place — pass a copy if needed).

    Returns
    -------
    np.ndarray
        Trace with NaNs replaced by interpolated values.
    """
    nan_idx = np.where(np.isnan(trace))[0]
    valid_idx = np.where(~np.isnan(trace))[0]
    trace[nan_idx] = np.interp(nan_idx, valid_idx, trace[valid_idx])
    return trace


def find_nearest(array, value):
    """Return the index of the element in array closest to value.

    Parameters
    ----------
    array : array-like
    value : float

    Returns
    -------
    int
        Index of the nearest element.
    """
    array = np.asarray(array)
    diffs = np.abs(array - value)
    return int(np.nanargmin(diffs))


# ---------------------------------------------------------------------------
# 2. Signal preprocessing
# ---------------------------------------------------------------------------

def lowpass_butter(input_signal, fps, cutoff, order=5, linear_interp=True):
    """Apply a zero-phase Butterworth lowpass filter.

    NaN values are linearly interpolated before filtering.

    Parameters
    ----------
    input_signal : array-like
    fps : float
        Sampling rate in Hz.
    cutoff : float
        Cutoff frequency in Hz.
    order : int
        Filter order (default 5).
    linear_interp : bool
        If True, interpolate NaNs before filtering. If False, raises an
        exception when NaNs are present.

    Returns
    -------
    np.ndarray
        Filtered signal.
    """
    input_signal = np.array(input_signal, dtype=float)

    if np.any(np.isnan(input_signal)):
        if not linear_interp:
            raise ValueError(
                "NaN values detected. Set linear_interp=True to interpolate them."
            )
        nan_idx = np.where(np.isnan(input_signal))[0]
        valid_idx = np.where(~np.isnan(input_signal))[0]
        input_signal[nan_idx] = np.interp(nan_idx, valid_idx, input_signal[valid_idx])

    sos = signal.butter(order, cutoff, btype="lowpass", output="sos", fs=fps)
    return signal.sosfiltfilt(sos, input_signal)

def lowpass_hamming(signal, signal_freq, lowpass_freq=1.0):
    """Zero-phase Hamming lowpass filter with reflection padding.
    Replicates the 1 Hz Hamming lowpass filter used by the lab's
    pupil tracking pipeline (FilterMethod._lowpass_hamming)."""
    hamming_length = int(2 * round(signal_freq / lowpass_freq, 0) + 1)
    hamming_filter = hamming(hamming_length, sym=True)
    hamming_filter = hamming_filter / np.sum(hamming_filter)
    pad_length = len(hamming_filter)
    padded = np.pad(signal, pad_length, mode='reflect')
    filtered = convolve(padded, hamming_filter, mode='same')
    return filtered[pad_length:-pad_length]


def median_filter(sig, signal_freq, window_sec):
    """Apply a median filter with reflection padding to avoid edge artifacts.

    Parameters
    ----------
    sig : array-like
    signal_freq : float
        Sampling rate in Hz.
    window_sec : float
        Filter window length in seconds.

    Returns
    -------
    np.ndarray
        Median-filtered signal.
    """
    window_idx = int(round(signal_freq * window_sec))
    if window_idx % 2 == 0:
        window_idx += 1  # kernel size must be odd

    padded = np.pad(sig, window_idx, mode="reflect")
    filtered = medfilt(padded, window_idx)
    return filtered[window_idx:-window_idx]


def detrend_polynomial(sig, order=4):
    """Remove a polynomial trend from a signal, preserving the median offset.

    Parameters
    ----------
    sig : array-like
    order : int
        Polynomial order (default 4).

    Returns
    -------
    np.ndarray
        Detrended signal.
    """
    sig = np.asarray(sig, dtype=float)
    frame_nums = np.arange(len(sig))
    trendline = np.polyval(np.polyfit(frame_nums, sig, order), frame_nums)
    return sig - trendline + np.nanmedian(sig)


def rolling_percentile(sig, percentile, fps, window_in_sec=20 * 60, use_center=True):
    """Compute a rolling percentile baseline with reflection padding.

    Parameters
    ----------
    sig : array-like
    percentile : float
        Percentile expressed as a decimal (e.g. 0.05 for 5th percentile).
    fps : float
        Sampling rate in Hz.
    window_in_sec : float
        Rolling window size in seconds (default 20 minutes).
    use_center : bool
        If True, the window is centered on each sample.

    Returns
    -------
    np.ndarray
        Rolling percentile baseline, same length as sig.
    """
    window_size = min(round(window_in_sec * fps), len(sig) * 2)
    extended = np.concatenate([sig[::-1], sig, sig[::-1]])
    baseline = (
        pd.Series(extended)
        .rolling(window_size, center=use_center)
        .quantile(percentile, interpolation="lower")
    )
    return np.array(baseline[len(sig): 2 * len(sig)])


def normalize_dFF(sig, fps, max_ratio):
    """Normalize a fluorescence trace to ΔF/F using a rolling 5th-percentile baseline.

    Parameters
    ----------
    sig : array-like
    fps : float
        Sampling rate in Hz.
    max_ratio : float
        Maximum allowed ratio of signal range to 5th percentile. Raises an
        exception if exceeded, to guard against traces where dF/F is unreliable.

    Returns
    -------
    np.ndarray
        dF/F normalized signal.

    Raises
    ------
    ValueError
        If the baseline is negative, or if the range/5th-percentile ratio
        exceeds max_ratio.
    """
    sig = np.asarray(sig, dtype=float)
    baseline = rolling_percentile(sig, 0.05, fps)

    rng = np.nanmax(sig) - np.nanmin(sig)
    ratio = abs(rng / np.percentile(sig, 5))
    if ratio > max_ratio:
        raise ValueError(
            f"Range-to-5th-percentile ratio ({ratio:.2f}) exceeds max_ratio "
            f"({max_ratio}). dF/F will not be computed."
        )
    if np.nanmin(baseline) < 0:
        raise ValueError("Baseline is negative; dF/F cannot be computed.")

    return (sig - baseline) / baseline


def normalize_0_1(trace):
    """Min-max normalize a trace to [0, 1].

    Parameters
    ----------
    trace : array-like

    Returns
    -------
    np.ndarray
    """
    trace = np.asarray(trace, dtype=float)
    return (trace - np.nanmin(trace)) / (np.nanmax(trace) - np.nanmin(trace))


def zscore(x):
    """Z-score a signal using its mean and standard deviation.

    Parameters
    ----------
    x : array-like

    Returns
    -------
    np.ndarray
    """
    x = np.asarray(x, dtype=float)
    return (x - np.nanmean(x)) / np.nanstd(x)


def safe_zscore(x):
    """Z-score a signal, returning zeros if the standard deviation is near zero.

    Parameters
    ----------
    x : array-like

    Returns
    -------
    np.ndarray
    """
    x = np.asarray(x, dtype=float)
    s = np.std(x)
    if s < 1e-8:
        return np.zeros_like(x)
    return (x - np.mean(x)) / s


def preprocess_dFF(trace, fps, max_ratio, filt=True, interp=True, **kwargs):
    """Full preprocessing pipeline: detrend → filter → dF/F normalize → resample.

    Parameters
    ----------
    trace : array-like
    fps : float
        Native sampling rate of the fluorescence trace (Hz).
    max_ratio : float
        Passed to normalize_dFF.
    filt : bool
        If True, apply lowpass filter. Requires kwarg ``lowpass_cutoff``.
    interp : bool
        If True, resample to new timestamps. Requires kwargs
        ``old_times`` and ``new_times``.

    Returns
    -------
    np.ndarray
        Preprocessed trace.
    """
    trace = detrend_polynomial(trace)
    if filt:
        trace = lowpass_butter(trace, fps, kwargs["lowpass_cutoff"])
    trace = normalize_dFF(trace, fps, max_ratio)
    if interp:
        trace = interpolate_to_new_timestamps(
            kwargs["old_times"], trace, kwargs["new_times"]
        )
    return trace


def preprocess_no_dFF(trace, fps, filt=True, interp=True, **kwargs):
    """Preprocessing pipeline without dF/F normalization: detrend → filter → resample.

    Parameters
    ----------
    trace : array-like
    fps : float
    filt : bool
    interp : bool

    Returns
    -------
    np.ndarray
    """
    trace = detrend_polynomial(trace)
    if filt:
        trace = lowpass_butter(trace, fps, kwargs["lowpass_cutoff"])
    if interp:
        trace = interpolate_to_new_timestamps(
            kwargs["old_times"], trace, kwargs["new_times"]
        )
    return trace


# ---------------------------------------------------------------------------
# 3. Hemodynamic correction
# ---------------------------------------------------------------------------

def correct_regression_no_runs(sensor, control, run_onsets, run_offsets, timestamps):
    """Subtract hemodynamic artifact from a fluorescence trace.

    Estimates the hemodynamic contribution by regressing the control
    (inert fluorophore) onto the sensor signal during non-running
    periods, then subtracts the scaled control from the sensor.

    Parameters
    ----------
    sensor : array-like
        NE sensor fluorescence trace (preprocessed, on ``timestamps`` grid).
    control : array-like
        Inert control fluorescence trace (preprocessed, on ``timestamps`` grid).
    run_onsets : array-like
        Run onset times in seconds.
    run_offsets : array-like
        Run offset times in seconds.
    timestamps : array-like
        Time axis (seconds) for sensor and control.

    Returns
    -------
    corrected_NE : np.ndarray
        Hemodynamic-corrected NE trace.
    sensor : np.ndarray
        Uncorrected sensor (passed through unchanged).
    control : np.ndarray
        Control trace (passed through unchanged).
    """
    sensor = np.array(sensor, dtype=float)
    control = np.array(control, dtype=float)

    # Mask running periods (plus a 5 s pre-pad and 15 s post-pad)
    sensor_no_runs = sensor.copy()
    control_no_runs = control.copy()
    for onset, offset in zip(run_onsets, run_offsets):
        start_idx = find_nearest(timestamps, onset - 5)
        stop_idx = find_nearest(timestamps, offset + 15)
        sensor_no_runs[start_idx:stop_idx] = np.nan
        control_no_runs[start_idx:stop_idx] = np.nan

    # Estimate scaling factor β via linear regression on quiet periods
    valid = ~np.isnan(sensor_no_runs) & ~np.isnan(control_no_runs)
    beta = np.polyfit(control_no_runs[valid], sensor_no_runs[valid], 1)[0]

    # Subtract scaled control
    corrected_control = control + (
        np.nanmean(sensor_no_runs) - np.nanmean(control_no_runs)
    )
    corrected_NE = (
        sensor - beta * corrected_control + np.nanmedian(sensor_no_runs)
    )

    return corrected_NE, sensor, control


# ---------------------------------------------------------------------------
# 4. Analysis helpers
# ---------------------------------------------------------------------------

def standard_error(all_traces, avg_trace):
    """Compute standard error bounds for a collection of traces.

    Handles NaNs by using the count of finite values per time point.

    Parameters
    ----------
    all_traces : array-like, shape (n_traces, n_timepoints)
    avg_trace : array-like, shape (n_timepoints,)

    Returns
    -------
    lower_y : np.ndarray
    upper_y : np.ndarray
    """
    all_traces = np.array(all_traces)
    num = np.nanstd(all_traces, axis=0)
    den = np.array([np.sqrt(np.sum(np.isfinite(col))) for col in all_traces.T])
    se = np.divide(num, den)
    return avg_trace - se, avg_trace + se


def get_run_psth(
    run_onsets, run_offsets, fluorescence, treadmill,
    onset_pre, onset_post, offset_pre, offset_post
):
    """Extract peri-stimulus time histograms (PSTHs) aligned to run onsets/offsets.

    Parameters
    ----------
    run_onsets : list of int
        Sample indices of run onsets.
    run_offsets : list of int
        Sample indices of run offsets.
    fluorescence : array-like
        Fluorescence trace on the same time grid as ``treadmill``.
    treadmill : array-like
        Treadmill velocity trace.
    onset_pre, onset_post : int
        Samples before/after each run onset to include.
    offset_pre, offset_post : int
        Samples before/after each run offset to include.

    Returns
    -------
    onset_bx, onset_fl, offset_bx, offset_fl : list of np.ndarray
        Treadmill and fluorescence snippets aligned to run onsets and offsets.
        Fluorescence snippets are baseline-subtracted (mean of pre-onset window)
        and converted to percent change.
    """
    onset_bx, onset_fl, offset_bx, offset_fl = [], [], [], []
    onset_baseline = None  # shared across onset/offset pairs

    for onset, offset in zip(run_onsets, run_offsets):
        # --- onset snippet ---
        bx = treadmill[onset - onset_pre: onset + onset_post]
        fl = fluorescence[onset - onset_pre: onset + onset_post]
        if len(bx) == onset_pre + onset_post:
            onset_bx.append(bx)
            onset_baseline = np.nanmean(fl[:onset_pre])
            onset_fl.append((fl - onset_baseline) * 100)

        # --- offset snippet ---
        bx = treadmill[offset - offset_pre: offset + offset_post]
        fl = fluorescence[offset - offset_pre: offset + offset_post]
        if len(bx) == offset_pre + offset_post and onset_baseline is not None:
            offset_bx.append(bx)
            offset_fl.append((fl - onset_baseline) * 100)

    return onset_bx, onset_fl, offset_bx, offset_fl
