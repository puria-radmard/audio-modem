from scipy import signal
import matplotlib.pyplot as plt
import numpy as np
import sounddevice as sd
from scipy import fft, ifft
from tempfile import TemporaryFile
import sys

outfile = TemporaryFile()
fs=44100
duration = 10 # seconds
myrecording = sd.rec(int(duration * fs), samplerate=fs, channels=1)
sd.wait()
myrecording = np.array(myrecording)
np.save(sys.argv[1], myrecording)
myrecording = myrecording.reshape(-1)
fft_out= fft(myrecording)