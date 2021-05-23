Main classes:

1. Channel - for transmitting audio when using a simulated channel, and for storing and allowing updating of the estimated impulse/spectrum of real channels
2. Synchronisation - for inserting chirps and pilot tones (returned to later) at transmission; and (matched) filtering and extracting pilot tones at reception
3. Modulation & Demodulation - the classes modulating bits into constellation values and constellation values into OFDM symbols, and the reverse, respectively
4. Estimation - a child class of Demodulation that estimates channel impulse/spectrum. This was originally only through random OFDM symbols, but later pilot tone estimation was adde
5. Transmitter & Receiver - child classes of Modulation and Estimation respectively, which act as wrappers for the whole data transmission pipeline, from bit modulation, to recording, to demodulation back to bits