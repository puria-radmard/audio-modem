from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np


def constellation_plot(const_hist):
    fig = plt.figure()
    arr = np.array(const_hist)
    plt.scatter(arr.real, arr.imag)
    fig.savefig("received constallation bits", dpi=fig.dpi)


def chirp_frame_location_plot(channel_output, chirp_filtered, chirp_slices):
    fig, axs = plt.subplots(3)
    axs[0].plot(channel_output)
    axs[1].plot(chirp_filtered)

    for csl in chirp_slices:
        axs[0].plot([csl.start, csl.stop], [0, 0])

    m = max(chirp_filtered)
    a = np.argmax(chirp_filtered)
    axs[2].scatter(range(len(chirp_filtered)), chirp_filtered)
    axs[2].set_xlim(0.8 * a, 1.2 * a)
    axs[2].set_ylim(0.95 * m, 1.05 * m)

    fig.savefig("segmented chirp frames")
