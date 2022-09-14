import numpy as np
import emd
from matplotlib import pyplot as plt


def plot_time_domain(txy, txy_refined, title='time domain positions'):
    fig = plt.figure()
    plt.title(title)
    plt.xlabel('time (s)')
    plt.ylabel('position (px)')
    plt.plot(txy[:, 0], txy[:, 1], '--', label='x int px')
    plt.plot(txy[:, 0], txy[:, 2], '--', label='y int px')
    plt.plot(txy_refined[:, 0], txy_refined[:, 1], label='x sub px')
    plt.plot(txy_refined[:, 0], txy_refined[:, 2], label='y sub px')
    plt.legend()
    return fig


def plot_frequency_domain(txy, txy_refined, title='frequency domain displacements'):
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
    fft_f = np.fft.fftfreq(x.shape[0], d=delta_t)
    n_bins = len(x)
    plt.plot(fft_f[0:n_bins], np.abs(pxx)[0:n_bins], '--', label='x int px')
    plt.plot(fft_f[0:n_bins], np.abs(pyy)[0:n_bins], '--', label='y int px')
    plt.plot(fft_f[0:n_bins], np.abs(pxx_refined)[0:n_bins], label='x sub px')
    plt.plot(fft_f[0:n_bins], np.abs(pyy_refined)[0:n_bins], label='y sub px')
    plt.title(title)
    plt.xlabel('frequency (Hz)')
    plt.ylabel('position (px / rad)')
    plt.xlim((0, 0.5/delta_t))
    plt.legend()
    return fig


def plot_emd(txy, txy_refined, title='empirical mode decomposition of positions'):
    fig = plt.figure()
    delta_ts = np.diff(txy[:, 0])
    delta_t = np.round(delta_ts.mean(), 6)
    assert (np.max(delta_ts) - np.min(delta_ts)) / delta_t < 1E-6, \
        f'Problem with sample time. {np.max(delta_ts)=}, {np.min(delta_ts)=}, {np.mean(delta_ts)=}'
    x = txy[:, 1]
    y = txy[:, 2]
    x_refined = txy_refined[:, 1]
    y_refined = txy_refined[:, 2]
    n_bins = len(x)
    imf_x = emd.sift.sift(x, max_imfs=2)
    # imf_y = emd.sift.sift(y)
    # imf_x_refined = emd.sift.sift(x_refined)
    # imf_y_refined = emd.sift.sift(y_refined)
    emd.plotting.plot_imfs(imf_x[:n_bins, :])
    plt.title(title)
    return fig
