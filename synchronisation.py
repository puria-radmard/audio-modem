import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import chirp
from scipy.io.wavfile import write
import pyaudio

def chirp_time_domain(t, omega_1, omega_2):
    T = len(t)
    l = np.log(omega_1/omega_2)
    brackets = np.e**((t/T)*l)
    inner = (omega_1*T)/(l)
    return np.sin(inner*(brackets - 1))

T = 10
fs= 44100
_t = np.linspace(0, T, fs*T)
samples = chirp(_t, 60, T, 5000)
plt.plot(_t, samples)
plt.show()

p = pyaudio.PyAudio()
stream = p.open(format=pyaudio.paFloat32,
                channels=2,
                rate=44100,
                frames_per_buffer=1024,
                output=True,
                )
stream.write(samples.astype(np.float32).tostring())
stream.close()

print(len(samples))
padded_zeros = np.zeros(500000)
padded_chirp = np.concatenate((padded_zeros, samples ,padded_zeros))
print(len(padded_chirp))
scaled = np.int16(padded_chirp/np.max(np.abs(padded_chirp)) * 32767)
write('paddedCHIRP.wav', 44100, scaled)