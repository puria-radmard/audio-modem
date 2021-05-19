import numpy as np
import sounddevice as sd

arr = np.load("test.npz.npy")
sd.play(arr, 44100)
sd.wait()