import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import chirp
import sounddevice as sd
from scipy.signal import convolve

def chirp_time_domain(t, omega_1, omega_2):
    T = len(t)
    l = np.log(omega_1/omega_2)
    brackets = np.e**((t/T)*l)
    inner = (omega_1*T)/(l)
    return np.sin(inner*(brackets - 1))

T = 10
fs= 44100
_t = np.linspace(0, T, fs*T)
check_signal = chirp(_t, 60, T, 5000)


fs=44100
duration = 1# 40 # seconds
myrecording = sd.rec(int(duration * fs), samplerate=fs, channels=1)
sd.wait()
print("recording done")
myrecording = np.array(myrecording)
myrecording = myrecording.reshape(-1)
matched_filter_output = convolve(myrecording, check_signal[::-1], method = 'fft')

plt.plot(matched_filter_output)
plt.savefig("matched_filter_output.png")
import pdb; pdb.set_trace()
