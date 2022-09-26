import numpy as np
from utils.kalman_filter import kf_constvel_smoother
import emd
from matplotlib import pyplot as plt


def plot_time_domain(txy, txy_refined, txy_smoothed=None, title='time domain positions'):
    fig, ax = plt.subplots(txy.shape[1] - 1, 1, sharex='all')
    ax[0].set_title(title)
    ax[-1].set_xlabel('time (s)')
    ax[0].set_ylabel('x position')
    ax[1].set_ylabel('y position')
    if txy.shape[1] == 4:
        ax[2].set_ylabel('angle (°)')
    ax[0].plot(txy[:, 0], txy[:, 1], ':', label='x int px')
    ax[1].plot(txy[:, 0], txy[:, 2], ':', label='y int px')
    if txy.shape[1] == 4:
        plt.plot(txy[:, 0], txy[:, 3], ':', label='angle int deg')
    ax[0].plot(txy_refined[:, 0], txy_refined[:, 1], '--', label='x sub px')
    ax[1].plot(txy_refined[:, 0], txy_refined[:, 2], '--', label='y sub px')
    if txy.shape[1] == 4:
        ax[2].plot(txy_refined[:, 0], txy_refined[:, 3], '--', label='angle sub deg')
    if txy_smoothed is not None:
        ax[0].plot(txy_smoothed[:, 0], txy_smoothed[:, 1], label='smoothed x sub px')
        ax[1].plot(txy_smoothed[:, 0], txy_smoothed[:, 2], label='smoothed y sub px')
        if txy.shape[1] == 4:
            ax[2].plot(txy_smoothed[:, 0], txy_smoothed[:, 3], label='smoothed angle sub deg')
    [ax[i].legend() for i in range(len(ax))]
    return fig


def plot_frequency_domain(txy, txy_refined, txy_smoothed, title='frequency domain displacements'):
    fig = plt.figure()
    delta_ts = np.diff(txy[:, 0])
    delta_t = np.round(delta_ts.mean(), 6)
    assert (np.max(delta_ts) - np.min(delta_ts)) / delta_t < 1E-6, \
        f'Problem with sample time. {np.max(delta_ts)=}, {np.min(delta_ts)=}, {np.mean(delta_ts)=}'
    x = txy[:, 1] - txy[:, 1].mean()
    y = txy[:, 2] - txy[:, 2].mean()
    pxx = np.fft.fft(x)
    pyy = np.fft.fft(y)
    x_refined = txy_refined[:, 1] - txy_refined[:, 1].mean()
    y_refined = txy_refined[:, 2] - txy_refined[:, 2].mean()
    pxx_refined = np.fft.fft(x_refined)
    pyy_refined = np.fft.fft(y_refined)
    x_smoothed = txy_smoothed[:, 1] - txy_smoothed[:, 1].mean()
    y_smoothed = txy_smoothed[:, 2] - txy_smoothed[:, 2].mean()
    pxx_smoothed = np.fft.fft(x_smoothed)
    pyy_smoothed = np.fft.fft(y_smoothed)
    fft_f = np.fft.fftfreq(x.shape[0], d=delta_t)
    n_bins = len(x)//2
    plt.plot(fft_f[0:n_bins], np.abs(pxx)[0:n_bins], '--', label='x int px')
    plt.plot(fft_f[0:n_bins], np.abs(pyy)[0:n_bins], '--', label='y int px')
    plt.plot(fft_f[0:n_bins], np.abs(pxx_refined)[0:n_bins], label='x sub px')
    plt.plot(fft_f[0:n_bins], np.abs(pyy_refined)[0:n_bins], label='y sub px')
    plt.plot(fft_f[0:n_bins], np.abs(pxx_smoothed)[0:n_bins], label='smoothed x sub px')
    plt.plot(fft_f[0:n_bins], np.abs(pyy_smoothed)[0:n_bins], label='smoothed y sub px')
    plt.title(title)
    plt.xlabel('frequency (Hz)')
    plt.ylabel('position (px / rad)')
    plt.xlim((0, 0.5/delta_t))
    plt.legend()
    return fig


def perform_kalman_filter(txy_refined, kalman_param):
    txy_smoothed = txy_refined.copy()
    delta_ts = np.diff(txy_smoothed[:, 0])
    delta_t = np.round(delta_ts.mean(), 6)
    assert (np.max(delta_ts) - np.min(delta_ts)) / delta_t < 1E-6, \
        f'Problem with sample time. {np.max(delta_ts)=}, {np.min(delta_ts)=}, {np.mean(delta_ts)=}'

    for i in [1, txy_smoothed.shape[1] - 1]:
        signal = txy_refined[:, i].copy()
        velocity_std = signal.std() * 2 * np.pi * kalman_param['freq_of_interest']
        txy_smoothed[:, i] = kf_constvel_smoother(signal, dt=delta_t,
                                                  measurement_error_std=kalman_param['measurement_error_std'],
                                                  velocity_std=velocity_std)
    return txy_smoothed


def plot_emd(txy, mask_freqs, max_imfs=4, title='empirical mode decomposition of positions'):
    delta_ts = np.diff(txy[:, 0])
    delta_t = np.round(delta_ts.mean(), 6)
    assert (np.max(delta_ts) - np.min(delta_ts)) / delta_t < 1E-6, \
        f'Problem with sample time. {np.max(delta_ts)=}, {np.min(delta_ts)=}, {np.mean(delta_ts)=}'
    x = (txy[:, 1] - txy[:, 1].mean())/(np.abs(txy[:, 1]).max())
    y = (txy[:, 2] - txy[:, 2].mean())/(np.abs(txy[:, 2]).max())
    imf_x, mask_freqs = emd.sift.mask_sift(x, mask_freqs=delta_t*np.array(mask_freqs), ret_mask_freq=True,
                                           max_imfs=max_imfs)
    print(mask_freqs/delta_t)
    # imf_y = emd.sift.sift(y)
    emd.plotting.plot_imfs(imf_x, X=x, sample_rate=1/delta_t, xlabel='time (s)', sharey=False)
    plt.title(title)
    return None
