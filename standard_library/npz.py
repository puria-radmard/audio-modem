import numpy as np
import sounddevice as sd

arr = np.load("output.npy")
sd.play(arr, 44100)
sd.wait()