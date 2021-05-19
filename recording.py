from scipy import signal
import matplotlib.pyplot as plt
import numpy as np
import sounddevice as sd
from scipy import fft, ifft
from tempfile import TemporaryFile

outfile = TemporaryFile()
noise = np.random.normal(0,1,44100*10)
fs=44100
duration = 10 # seconds
myrecording = sd.rec(int(duration * fs), samplerate=fs, channels=1)
sd.wait()
myrecording = np.array(myrecording)
myrecording = myrecording.reshape(-1)
fft_out= fft(myrecording)

import pdb;pdb.set_trace()

plt.figure()
plt.plot(f, fft_out)
plt.ylim([1e-4, 1e1])
plt.xlim(0,2000)
plt.xlabel('frequency [Hz]')
plt.ylabel('Linear spectrum [V RMS]')
plt.show()