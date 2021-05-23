from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import os, cv2
import glob
import matplotlib.animation as animation


def generate_constellation_video(folder, img, animation_name):

    frames = []

    for i in range(len(img)):
        arr = img[i]        
        fig, axs = plt.subplots(1)
        axs.set_xlim(-2, 2)
        axs.set_ylim(-2, 2)
        axs.scatter(arr.real, arr.imag)
        fig.savefig(folder + "/file%02d" % i)
        frames.append(folder + "/file%02d.png" % i)
    
    frame = cv2.imread(frames[0])
    height, width, layers = frame.shape
    video = cv2.VideoWriter(f"{folder}/{animation_name}.avi", cv2.VideoWriter_fourcc(*'XVID'), 1.5, (width,height)) 

    for image in frames:
        video.write(cv2.imread(image))

    cv2.destroyAllWindows()
    video.release()

    for file_name in glob.glob(f"{folder}/file*.png"):
        os.remove(file_name)


def generate_phaseshifting_video(folder, img, animation_name, pilot_symbol):

    frames = []

    for j, (left_pilot_idx, recovered_pilot_tones_left, phase_shifts, N) in enumerate(img):
        fig, axs = plt.subplots(2,figsize=(15,30))
        axs[0].set_ylim(-2, 0)
        axs[0].set_xlim(-2, 0)

        sc = axs[0].scatter(x = recovered_pilot_tones_left.real, y = recovered_pilot_tones_left.imag, c = left_pilot_idx)
        cbar = plt.colorbar(sc)
        cbar.set_label(f"Index in OFDM symbol (N = {N})")

        axs[1].plot(left_pilot_idx, phase_shifts)
    
        fig.savefig(f"{folder}/pilotestimationrotation{j}.png")
        frames.append(f"{folder}/pilotestimationrotation{j}.png")

    frame = cv2.imread(frames[0])
    height, width, layers = frame.shape
    video = cv2.VideoWriter(f"{folder}/{animation_name}.avi", cv2.VideoWriter_fourcc(*'XVID'), 1.5, (width,height)) 

    for image in frames:
        video.write(cv2.imread(image))

    cv2.destroyAllWindows()
    video.release()

    for file_name in glob.glob(f"{folder}/pilotestimationrotation*.png"):
        pass#os.remove(file_name)
    

def constellation_plot(const_hist):
    fig, axs = plt.subplots(1)
    arr = np.array(const_hist)
    axs.scatter(arr.real, arr.imag)
    axs.set_xlim(-2, 2)
    axs.set_ylim(-2, 2)
    fig.savefig("received constallation bits", dpi=fig.dpi)
    return arr


def chirp_frame_location_plot(channel_output, chirp_filtered, chirp_slices):

    fig, axs = plt.subplots(3)
    axs[0].plot(channel_output)
    real_stop = min([chirp_slices[0].stop, len(channel_output)])
    axs[0].plot(range(chirp_slices[0].start, real_stop), channel_output[chirp_slices[0]])

    axs[1].plot(chirp_filtered)

    for csl in chirp_slices:
        axs[0].plot([csl.start, csl.stop], [0, 0])

    m = max(chirp_filtered)
    a = np.argmax(chirp_filtered)
    axs[2].scatter(range(len(chirp_filtered)), chirp_filtered)
    axs[2].set_xlim(0.8 * a, 1.2 * a)
    axs[2].set_ylim(0.95 * m, 1.05 * m)

    fig.savefig("segmented chirp frames")
