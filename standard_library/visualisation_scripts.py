from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np

def constellation_plot(const_hist):
    fig = plt.figure()
    arr = np.array(const_hist)
    plt.scatter(arr.real, arr.imag)
    fig.savefig('received constallation bits', dpi=fig.dpi)

def chirp_frame_location_plot(channel_output, chirp_filtered, chirp_slices):
    fig, axs = plt.subplots(2)
    axs[0].plot(channel_output)
    axs[1].plot(chirp_filtered)

    for csl in chirp_slices:
        axs[0].plot([csl.start, csl.stop], [0, 0])
    fig.savefig('segmented chirp frames')