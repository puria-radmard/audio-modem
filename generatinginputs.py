import wave
import numpy as np
import sounddevice as sd
import matplotlib.pyplot as plt
import numpy as np

# Generating Impulse
data = [128 for i in range(100000)] # zeroes
for j in range(50000,50100):
    data[j] = 255

# Generating train of impulse
datalong = data * 10
padded_zeros_front = [128 for i in range(500000)]
padded_impulse1 = padded_zeros_front + datalong
padded_impulse = padded_impulse1 + padded_zeros_front
datalongpadded = bytes(padded_impulse)

with wave.open(r'paddedIMPULSE.wav', 'wb') as f:
    f.setnchannels(1) # mono
    f.setsampwidth(1) 
    f.setframerate(44100) # standard sample rate
    f.writeframes(datalongpadded)

    
# Generating White Noise
noise = np.random.normal(0,1,44100*10)
sd.play(noise, 44100)

from scipy.io.wavfile import write
print(len(noise))
padded_zeros = np.zeros(500000)
padded_noise = np.concatenate((padded_zeros, noise ,padded_zeros))
print(len(padded_noise))
scaled = np.int16(padded_noise/np.max(np.abs(padded_noise)) * 32767)
write('paddedNOISE.wav', 44100, scaled)



# Generating Chirp
from scipy.signal import chirp

T = 10
fs= 44100
_t = np.linspace(0, T, fs*T)
samples = chirp(_t, 60, T, 5000)

plt.plot(_t, samples)
plt.show()

import pyaudio
p = pyaudio.PyAudio()
stream = p.open(format=pyaudio.paFloat32,
                channels=2,
                rate=44100,
                frames_per_buffer=1024,
                output=True,
                )
stream.write(samples.astype(np.float32).tostring())
stream.close()

from scipy.io.wavfile import write
print(len(samples))
padded_zeros = np.zeros(500000)
padded_chirp = np.concatenate((padded_zeros, samples ,padded_zeros))
print(len(padded_chirp))
scaled = np.int16(padded_chirp/np.max(np.abs(padded_chirp)) * 32767)
write('paddedCHIRP.wav', 44100, scaled)
