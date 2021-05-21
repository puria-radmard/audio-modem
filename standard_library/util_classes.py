import numpy as np
import pandas as pd
from scipy.signal import convolve
from scipy.io.wavfile import write
from typing import FrozenSet, List, Tuple
from scipy.fft import fft, ifft
from scipy.signal.ltisys import impulse
from scipy.signal import chirp
from util_objects import *
import binascii
import sys
import sounddevice as sd
from visualisation_scripts import *
from scipy.io.wavfile import read

class Channel:
    def __init__(self, impulse_response: np.ndarray):
        self.impulse_response = impulse_response

    def transmit(self, signal: np.ndarray) -> np.ndarray:
        # Transmit sequence by convolving with impulse reponse
        echo = convolve(signal, self.impulse_response)[:len(signal)]
        # echo += np.random.randn(len(signal))*0.0001
        return echo

    def transfer_function(self, length: int) -> np.ndarray:
        return fft(self.impulse_response, length)


class Synchronisation:
    def __init__(self, modes, N, L, chirp_func = None, chirp_length = None, num_OFDM_symbols_chirp = None, pilot_idx = None, pilot_symbol = None):
        self.modes = modes
        self.N = N
        self.L = L
        if "chirp" in modes:
            self.chirp_func = chirp_func
            self.num_OFDM_symbols_chirp = num_OFDM_symbols_chirp
            self.configure_chirp(chirp_length)
        if "pilot" in modes:
            self.pilot_idx = pilot_idx
            self.num_pilots = len(pilot_idx)
            self.pilot_symbol = pilot_symbol

    def configure_chirp(self, chirp_length):
        self.chirp_samples = int(chirp_length * fs)

        t = np.linspace(0, chirp_length, self.chirp_samples)
        chirp_arr = self.chirp_func(t)
        self.chirp = chirp_arr.reshape(-1)
        self.reversed_chirp = self.chirp[::-1]

        padded_chirp = np.concatenate([np.zeros(self.chirp_samples), self.chirp, np.zeros(self.chirp_samples)])
        auto_conv = convolve(padded_chirp, self.reversed_chirp)[:len(padded_chirp)]
        peak_sample_idx = np.argmax(auto_conv)
        self.delay_after_peak = 2*self.chirp_samples - peak_sample_idx

        self.total_post_peak_samples = self.delay_after_peak + self.num_OFDM_symbols_chirp * (N+L)

    def matched_filter(self, data):
        return convolve(data, self.reversed_chirp)[:len(data)]

    def locate_first_chirp(self, data):
        highest_sample = np.max(data)
        threshold = 0.8*highest_sample
        first_chirp_region_start_idx = np.argmax(data>threshold)
        first_chirp_region = slice(first_chirp_region_start_idx, first_chirp_region_start_idx + self.total_post_peak_samples)
        first_peak_idx = np.argmax(data[first_chirp_region]) + first_chirp_region_start_idx
        return first_peak_idx

    def locate_chirp_slices(self, data):
        frames_removed = 0
        _data = data.copy()
        slices_located = []
        
        while tqdm(len(_data) > 0):#(self.total_post_peak_samples):
            earliest_chirp_idx = self.locate_first_chirp(_data)
            earliest_frame_start = earliest_chirp_idx + self.delay_after_peak
            earliest_frame_end = earliest_chirp_idx + self.total_post_peak_samples

            slices_located.append(slice(earliest_frame_start+frames_removed, earliest_frame_end+frames_removed))

            cutout_samples = earliest_frame_end# - self.chirp_samples
            frames_removed += cutout_samples
            _data = data[frames_removed:]

        return slices_located        
    
    def insert_pilot_tones(self, constallation_values):

        if "pilot" in self.modes:
            post_pilot_constallation_values = []
            for j, c in enumerate(constallation_values):
                if j not in self.pilot_idx:
                    post_pilot_constallation_values.extend([c])
                else:
                    post_pilot_constallation_values.extend([self.pilot_symbol])
                    post_pilot_constallation_values.extend([c])
        else:
            post_pilot_constallation_values = constallation_values

        return post_pilot_constallation_values


class Modulation:
    def __init__(self, constellation_name: str, N: int, L: int):
        self.constellation = CONSTELLATIONS_DICT[constellation_name]
        self.constellation_length = len(list(self.constellation.keys())[0])
        self.N = N
        self.L = L

    def bits2constellation(self, bitstring: str) -> float:
        # Convert a single string of self.constellation length to the correct constellation value
        return self.constellation[tuple(int(b) for b in bitstring)]

    def sequence2constellation(self, bitstring: str) -> List[float]:
        # Convert a string of bits to a list of constellation value
        n = self.constellation_length
        frames = [bitstring[i : i + n] for i in range(0, len(bitstring), n)]
        if len(frames[-1]) < n:
            frames[-1] = frames[-1] + "0" * (n - len(frames[-1]))
        return [self.bits2constellation(frame) for frame in frames]

    def bits2OFDM(self, bitstring: str, synchroniser: Synchronisation) -> List[float]:

        constallation_values = self.sequence2constellation(bitstring)
        post_pilot_constallation_values = synchroniser.insert_pilot_tones(constallation_values)
        symmetric_conj = np.conj(post_pilot_constallation_values)[::-1]
        symmetric_conj_mirrored = np.concatenate([[0], post_pilot_constallation_values, [0], symmetric_conj])
        time_domain_symmetric_conj = ifft(symmetric_conj_mirrored)
        message_with_cyclical_prefix = np.concatenate([time_domain_symmetric_conj[-self.L:], time_domain_symmetric_conj])

        return message_with_cyclical_prefix.real

    @staticmethod
    def split_bitstring_into_chunks(data, N_mod):
        frames = [data[i : i + N_mod] for i in range(0, len(data), N_mod)]
        if len(frames[-1]) < N_mod:
            try:
                frames[-1] = frames[-1] + "0" * (N_mod - len(frames[-1]))
            except:
                frames[-1] = np.concatenate([frames[-1], [0] * (N_mod - len(frames[-1]))])
        return frames

    def data2OFDM(self, bitstring: str, synchroniser: Synchronisation, return_frames = False) -> List[float]:

        information_frequency_bins = (self.N/2) - 1

        constellation_per_chunk = information_frequency_bins if "pilot" not in synchroniser.modes else information_frequency_bins - synchroniser.num_pilots # account for pilot
        bits_per_chunk = int(constellation_per_chunk * self.constellation_length) # number of constallation values
        OFDM_data = self.split_bitstring_into_chunks(bitstring, bits_per_chunk)
        OFDM_data = [self.bits2OFDM(OFDM_symbol, synchroniser) for OFDM_symbol in OFDM_data]
        
        transmission_data = []
        for j, OFDM_symbol in enumerate(OFDM_data):
            if "chirp" in synchroniser.modes and j%synchroniser.num_OFDM_symbols_chirp == 0:
                transmission_data.extend(synchroniser.chirp)
            transmission_data.extend(OFDM_symbol)

        if not return_frames:
            return transmission_data
        else:
            return transmission_data, OFDM_data

    @staticmethod
    def publish_data(data, name):
        np.save(name, data)


class Demodulation:
    def __init__(self, N: int, L: int, constellation_name: str):
        self.N = N
        self.L = L
        self.constellation = CONSTELLATIONS_DICT[constellation_name]

    def OFDM2constellation(self, channel_output: np.ndarray, channel: Channel):
        num_frames = len(channel_output) / (self.N + self.L)
        with_cyclic_frames = np.array_split(channel_output, num_frames)
        message_frames = [list(w[self.L:]) for w in with_cyclic_frames]
        fft_frames = [fft(m, self.N) for m in message_frames]
        
        channel_TF = channel.transfer_function(self.N)
        deconvolved_frames = [
            np.divide(r, channel_TF)[1:int(self.N/2)] for r in fft_frames
        ]
        return deconvolved_frames

    def constellation2bits_single(self, constellation_value: float) -> Tuple[int]:
        return min(
            self.constellation.keys(),
            key = lambda x: abs(constellation_value - self.constellation[x])
        )

    def constellation2bits_sequence(self, constellation_values: List[float], synchronisation, show = True) -> str:
        symbol_bits_sequence = []
        const_hist = []
        for c in constellation_values:
            for d in c:
                symbol_bits_sequence.extend(self.constellation2bits_single(d))
                const_hist.extend([d])
        output_bitstring = "".join([str(a) for a in symbol_bits_sequence])

        if show:
            constellation_plot(const_hist)

        return output_bitstring

    def bitstring2text(self, bitstring: str) -> str:
        output_bytes = [
            bitstring[i : i + 8] for i in range(0, len(bitstring), 8)
        ]
        output_bytes = bytearray([int(i, 2) for i in output_bytes])

        return output_bytes

    def receive_channel_output(self, channel_output: np.ndarray, synchronisation: Synchronisation):
        
        OFDM_frames = []

        if "chirp" in synchronisation.modes:
            chirp_filtered = synchronisation.matched_filter(channel_output)
            chirp_slices = synchronisation.locate_chirp_slices(chirp_filtered)
            chirp_frames = [channel_output[sl] for sl in chirp_slices]
            for frame in chirp_frames:
                new_frames = np.array_split(frame, len(frame)/(self.N + self.L))
                OFDM_frames.extend(new_frames)

            chirp_frame_location_plot(channel_output, chirp_filtered, chirp_slices)

            return OFDM_frames
        
        else:
            raise NotImplementedError("Need chirp for synchronisation for now!")

    def OFDM2text(self, channel_output: np.ndarray, channel: Channel, synchronisation: Synchronisation) -> str:
        
        OFDM_frames = self.receive_channel_output(channel_output, synchronisation)

        bitstring = ""
        for frame in OFDM_frames:
            deconvolved_frame = self.OFDM2constellation(frame, channel)
            bitstring += self.constellation2bits_sequence(deconvolved_frame, sync)

        return bitstring


class Estimation(Demodulation):

    def __init__(self, N: int, L: int, constellation_name: str):
        super().__init__(N, L, constellation_name)

    def OFDM_channel_estimation(self, channel_output, synchronisation, ground_truth_OFDM_frames):
        received_OFDM_frames = self.receive_channel_output(channel_output, synchronisation)
        transfer_function_trials = []
        for j, ground_truth_OFDM_frame in enumerate(ground_truth_OFDM_frames):
            output_spectrum = fft(received_OFDM_frames[j][self.L:])
            input_spectrum = fft(ground_truth_OFDM_frame[self.L:]).round()    # MAY NEED FIXING IF NOT ALWAYS USING INTEGERS
            transfer_function = np.divide(output_spectrum, input_spectrum)      
            
            # MIGHT NOT WORK FOR ODD N
            transfer_function[0] = 0
            transfer_function[int(self.N/2)] = 0

            transfer_function_trials.append(transfer_function)

        fig = plt.figure()
        average_transfer_function = np.mean(transfer_function_trials, 0)
        average_impulse = ifft(average_transfer_function)
        plt.plot(average_impulse.real)
        plt.title("Channel response via OFDM")
        plt.xlabel("Sample number")
        plt.ylabel("Impulse coeff")      

        fig.savefig("OFDM_estimation")
        


if __name__ == "__main__":

    channel_impulse = np.array(pd.read_csv("channel.csv", header=None)[0])
    artificial_channel_output = list(pd.read_csv("file1.csv", header=None)[0])
    text = b'''
        The Longest Text Ever An attempt at creating the longest wall of text ever written. Check out some other LTEs! Hello, everyone! This is the LONGEST TEXT EVER! I was inspired by the various other "longest texts ever" on the internet, and I wanted to make my own. So here it is! This is going to be a WORLD RECORD! This is actually my third attempt at doing this. The first time, I didn't save it. The second time, the Neocities editor crashed. Now I'm writing this in Notepad, then copying it into the Neocities editor instead of typing it directly in the Neocities editor to avoid crashing. It sucks that my past two attempts are gone now. Those actually got pretty long. Not the longest, but still pretty long. I hope this one won't get lost somehow. Anyways, let's talk about WAFFLES! I like waffles. Waffles are cool. Waffles is a funny word. There's a Teen Titans Go episode called "Waffles" where the word "Waffles" is said a hundred-something times. It's pretty annoying. There's also a Teen Titans Go episode about Pig Latin. Don't know what Pig Latin is? It's a language where you take all the consonants before the first vowel, move them to the end, and add '-ay' to the end. If the word begins with a vowel, you just add '-way' to the end. For example, "Waffles" becomes "Afflesway". I've been speaking Pig Latin fluently since the fourth grade, so it surprised me when I saw the episode for the first time. I speak Pig Latin with my sister sometimes. It's pretty fun. I like speaking it in public so that everyone around us gets confused. That's never actually happened before, but if it ever does, 'twill be pretty funny. By the way, "'twill" is a word I invented recently, and it's a contraction of "it will". I really hope it gains popularity in the near future, because "'twill" is WAY more fun than saying "it'll". "It'll" is too boring. Nobody likes boring. This is nowhere near being the longest text ever, but eventually it will be! I might still be writing this a decade later, who knows? But right now, it's not very long. But I'll just keep writing until it is the longest! Have you ever heard the song "Dau Dau" by Awesome Scampis? It's an amazing song. Look it up on YouTube! I play that song all the time around my sister! It drives her crazy, and I love it. Another way I like driving my sister crazy is by speaking my own made up language to her. She hates the languages I make! The only language that we both speak besides English is Pig Latin. I think you already knew that. Whatever. I think I'm gonna go for now. Bye! Hi, I'm back now. I'm gonna contribute more to this soon-to-be giant wall of text. I just realised I have a giant stuffed frog on my bed. I forgot his name. I'm pretty sure it was something stupid though. I think it was "FROG" in Morse Code or something. Morse Code is cool. I know a bit of it, but I'm not very good at it. I'm also not very good at French. I barely know anything in French, and my pronunciation probably sucks. But I'm learning it, at least. I'm also learning Esperanto. It's this language that was made up by some guy a long time ago to be the "universal language". A lot of people speak it. I am such a language nerd. Half of this text is probably gonna be about languages. But hey, as long as it's long! Ha, get it? As LONG as it's LONG? I'm so funny, right? No, I'm not. I should probably get some sleep. Goodnight! Hello, I'm back again. I basically have only two interests nowadays: languages and furries. What? Oh, sorry, I thought you knew I was a furry. Haha, oops. Anyway, yeah, I'm a furry, but since I'm a young furry, I can't really do as much as I would like to do in the fandom. When I'm older, I would like to have a fursuit, go to furry conventions, all that stuff. But for now I can only dream of that. Sorry you had to deal with me talking about furries, but I'm honestly very desperate for this to be the longest text ever. Last night I was watching nothing but fursuit unboxings. I think I need help. This one time, me and my mom were going to go to a furry Christmas party, but we didn't end up going because of the fact that there was alcohol on the premises, and that she didn't wanna have to be a mom dragging her son through a crowd of furries. Both of those reasons were understandable. Okay, hopefully I won't have to talk about furries anymore. I don't care if you're a furry reading this right now, I just don't wanna have to torture everyone else. I will no longer say the F word throughout the rest of this entire text. Of course, by the F word, I mean the one that I just used six times, not the one that you're probably thinking of which I have not used throughout this entire text. I just realised that next year will be 2020. That's crazy! It just feels so futuristic! It's also crazy that the 2010s decade is almost over. That decade brought be a lot of memories. In fact, it brought be almost all of my memories. It'll be sad to see it go. I'm gonna work on a series of video lessons for Toki Pona. I'll expain what Toki Pona is after I come back. Bye! I'm back now, and I decided not to do it on Toki Pona, since many other people have done Toki Pona video lessons already. I decided to do it on Viesa, my English code. Now, I shall explain what Toki Pona is. Toki Pona is a minimalist constructed language that has only ~120 words! That means you can learn it very quickly. I reccomend you learn it! It's pretty fun and easy! Anyway, yeah, I might finish my video about Viesa later. But for now, I'm gonna add more to this giant wall of text, because I want it to be the longest! It would be pretty cool to have a world record for the longest text ever. Not sure how famous I'll get from it, but it'll be cool nonetheless. Nonetheless. That's an interesting word. It's a combination of three entire words. That's pretty neat. Also, remember when I said that I said the F word six times throughout this text? I actually messed up there. I actually said it ten times (including the plural form). I'm such a liar! I struggled to spell the word "liar" there. I tried spelling it "lyer", then "lier". Then I remembered that it's "liar". At least I'm better at spelling than my sister. She's younger than me, so I guess it's understandable. "Understandable" is a pretty long word. Hey, I wonder what the most common word I've used so far in this text is. I checked, and appearantly it's "I", with 59 uses! The word "I" makes up 5% of the words this text! I would've thought "the" would be the most common, but "the" is only the second most used word, with 43 uses. "It" is the third most common, followed by "a" and "to". Congrats to those five words! If you're wondering what the least common word is, well, it's actually a tie between a bunch of words that are only used once, and I don't wanna have to list them all here. Remember when I talked about waffles near the beginning of this text? Well, I just put some waffles in the toaster, and I got reminded of the very beginnings of this longest text ever. Okay, that was literally yesterday, but I don't care. You can't see me right now, but I'm typing with my nose! Okay, I was not able to type the exclamation point with just my nose. I had to use my finger. But still, I typed all of that sentence with my nose! I'm not typing with my nose right now, because it takes too long, and I wanna get this text as long as possible quickly. I'm gonna take a break for now! Bye! Hi, I'm back again. My sister is beside me, watching me write in this endless wall of text. My sister has a new thing where she just says the word "poop" nonstop. I don't really like it. She also eats her own boogers. I'm not joking. She's gross like that. Also, remember when I said I put waffles in the toaster? Well, I forgot about those and I only ate them just now. Now my sister is just saying random numbers. Now she's saying that they're not random, they're the numbers being displayed on the microwave. Still, I don't know why she's doing that. Now she's making annoying clicking noises. Now she's saying that she's gonna watch Friends on three different devices. Why!?!?! Hi its me his sister. I'd like to say that all of that is not true. Max wants to make his own video but i wont let him because i need my phone for my alarm.POOP POOP POOP POOP LOL IM FUNNY. kjnbhhisdnhidfhdfhjsdjksdnjhdfhdfghdfghdfbhdfbcbhnidjsduhchyduhyduhdhcduhduhdcdhcdhjdnjdnhjsdjxnj Hey, I'm back. Sorry about my sister. I had to seize control of the LTE from her because she was doing keymash. Keymash is just effortless. She just went back to school. She comes home from school for her lunch break. I think I'm gonna go again. Bye! Hello, I'm back. Let's compare LTE's. This one is only 8593 characters long so far. Kenneth Iman's LTE is 21425 characters long. The Flaming-Chicken LTE (the original) is a whopping 203941 characters long! I think I'll be able to surpass Kenneth Iman's not long from now. But my goal is to surpass the Flaming-Chicken LTE. Actually, I just figured out that there's an LTE longer than the Flaming-Chicken LTE. It's Hermnerps LTE, which is only slightly longer than the Flaming-Chicken LTE, at 230634 characters. My goal is to surpass THAT. Then I'll be the world record holder, I think. But I'll still be writing this even after I achieve the world record, of course. One time, I printed an entire copy of the Bee Movie script for no reason. I heard someone else say they had three copies of the Bee Movie script in their backpack, and I got inspired. But I only made one copy because I didn't want to waste THAT much paper. I still wasted quite a bit of paper, though. Now I wanna see how this LTE compares to the Bee Movie script. Okay, I checked, an
        '''
    text_bits = bin(int(binascii.hexlify(text), 16))[2:] # remove '0b'

    N = 1024
    L = 32

    if sys.argv[1] == "demod":
        channel = Channel(channel_impulse)
        demodulation = Demodulation(N = N, L = L, constellation_name="gray")
        output_text = demodulation.OFDM2text(artificial_channel_output, channel)

    elif sys.argv[1] == "pilot_sync":
        sync = Synchronisation(["pilot"], pilot_idx=[100, 200, 300, 400, 500], pilot_symbol=-1-1j, N = N, L = L)
        modulator = Modulation(constellation_name="gray", N=N, L=L)
        demodulation = Demodulation(N = N, L = L, constellation_name="gray")
        channel = Channel(impulse_response = channel_impulse)

        OFDM_transmission = modulator.data2OFDM(bitstring=text_bits, synchroniser=sync)
        modulator.publish_data(OFDM_transmission, "asf")
        channel_output = channel.transmit(OFDM_transmission)
        
        output_text = demodulation.OFDM2text(channel_output, channel, sync)
        output_text = int(f'0b{output_text}', 2)

        output_text = binascii.unhexlify('%x' % output_text)
    
    elif sys.argv[1] == "chirp_sync":

        T = 1.5
        c_func = lambda t: chirp(t, f0=20000, f1=60, t1=T, method = 'logarithmic')
        sync = Synchronisation(["chirp"], chirp_length = T, chirp_func=c_func, N = N, L = L, num_OFDM_symbols_chirp = 80)

        modulator = Modulation(constellation_name="gray", N=N, L=L)
        demodulation = Demodulation(N = N, L = L, constellation_name="gray")
        channel = Channel(impulse_response = channel_impulse)

        OFDM_transmission = modulator.data2OFDM(bitstring=text_bits, synchroniser=sync)
        modulator.publish_data(OFDM_transmission, "asf")
        channel_output = channel.transmit(OFDM_transmission)
        
        output_text = demodulation.OFDM2text(channel_output, channel, sync)
        output_text = int(f'0b{output_text}', 2)

        output_text = binascii.unhexlify('%x' % output_text)
        print(output_text)

    elif sys.argv[1] == "OFDM_estimation":
        
        T = 1.5
        c_func = lambda t: chirp(t, f0=20000, f1=60, t1=T, method = 'logarithmic')
        sync = Synchronisation(["chirp"], chirp_length = T, chirp_func=c_func, N = N, L = L, num_OFDM_symbols_chirp = 80)

        modulator = Modulation(constellation_name="gray", N=N, L=L)
        estimator = Estimation(constellation_name="gray", N=N, L=L)
        demodulation = Demodulation(N = N, L = L, constellation_name="gray")
        channel = Channel(impulse_response = channel_impulse)
        

        OFDM_transmission, OFDM_data = modulator.data2OFDM(bitstring=text_bits, synchroniser=sync, return_frames=True)
        modulator.publish_data(OFDM_transmission, "asf")
        
        channel_output = np.load("output.npy").reshape(-1)
        #channel_output = channel.transmit(OFDM_transmission)
        
        estimator.OFDM_channel_estimation(channel_output, synchronisation = sync, ground_truth_OFDM_frames = OFDM_data)