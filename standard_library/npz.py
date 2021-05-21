import sys
import numpy as np
import sounddevice as sd
import sys

arr = np.load(f"{sys.argv[1]}.npy")
sd.play(arr, 44100)
sd.wait()
