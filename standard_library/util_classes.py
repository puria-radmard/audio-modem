import pdb
import numpy as np
from numpy.core.numeric import full
import pandas as pd
from scipy import interpolate
from scipy.interpolate.polyint import KroghInterpolator
from scipy.signal import convolve
from typing import FrozenSet, List, Tuple
from scipy.fft import fft, ifft
from scipy.signal.ltisys import impulse
from scipy.signal import chirp
from util_objects import *
import binascii
import sys
import sounddevice as sd
from visualisation_scripts import *
import random
from scipy.io.wavfile import read
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline
#import commpy.channelcoding as cc
from sklearn.linear_model import LinearRegression
import ldpc

class Channel:
    def __init__(self, impulse_response: np.ndarray):
        self.impulse_response = impulse_response
        self.past_spectra = [self.transfer_function(len(impulse_response))]
        self.past_impulses = [impulse_response]

    def transmit(self, signal: np.ndarray, noise = 0.01) -> np.ndarray:
        # Transmit sequence by convolving with impulse reponse
        echo = convolve(signal, self.impulse_response)[: len(signal)]
        echo += np.random.randn(len(signal)) * noise
        return echo

    def update_channel_impulse(self, new_impulse, new_weight):
        self.impulse_response *= (1-new_weight)
        self.impulse_response += new_weight * new_impulse

    def update_channel_spectrum(self, new_spectrum, new_weight):
        current_spectrum = self.transfer_function(len(new_spectrum))
        updated_spectrum = (1-new_weight)*current_spectrum + new_weight*new_spectrum
        new_impulse_response = ifft(updated_spectrum, len(self.impulse_response))
        self.impulse_response = new_impulse_response

        self.past_impulses.append(new_impulse_response)
        self.past_spectra.append(updated_spectrum)
        

    def transfer_function(self, length: int) -> np.ndarray:
        return fft(self.impulse_response, length)


class Synchronisation:
    def __init__(
        self,
        modes,
        N,
        L,
        chirp_func=None,
        chirp_length=None,
        num_OFDM_symbols_chirp=None,
        pilot_idx=None,
        pilot_symbol=None,
    ):
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

        padded_chirp = np.concatenate(
            [np.zeros(self.chirp_samples), self.chirp, np.zeros(self.chirp_samples)]
        )
        auto_conv = convolve(padded_chirp, self.reversed_chirp)[: len(padded_chirp)]
        peak_sample_idx = np.argmax(auto_conv)
        self.delay_after_peak = 2 * self.chirp_samples - peak_sample_idx

        self.total_post_peak_samples = (
            self.delay_after_peak + self.num_OFDM_symbols_chirp * (self.N + self.L)
        )

    def matched_filter(self, data):
        return convolve(data, self.reversed_chirp)[: len(data)]

    def locate_chirp_slices(self, data, sample_shift=0):
        earliest_chirp_idx = np.argmax(data)
        earliest_frame_start = earliest_chirp_idx + self.delay_after_peak
        earliest_frame_end = earliest_chirp_idx + self.total_post_peak_samples

        start = earliest_frame_start
        stop = earliest_frame_end
        new_slice = slice(max(0, start + sample_shift), stop + sample_shift)
        
        return [new_slice]


    def locate_first_chirp(self, data, threshold):
        # Not used
        if all(data < threshold):
            return None
        first_chirp_region_start_idx = np.argmax(data > threshold)
        first_chirp_region = slice(
            first_chirp_region_start_idx,
            first_chirp_region_start_idx + self.total_post_peak_samples,
        )
        first_peak_idx = (
            np.argmax(data[first_chirp_region]) + first_chirp_region_start_idx
        )
        return first_peak_idx

    def OLD_locate_chirp_slices(self, data, sample_shift=0):
        frames_removed = 0
        _data = data.copy()
        slices_located = []
        highest_sample = np.max(data)
        threshold = 0.8 * highest_sample

        while len(_data) > 0:  # (self.total_post_peak_samples):
            earliest_chirp_idx = self.locate_first_chirp(_data, threshold)
            if not earliest_chirp_idx:
                break
            earliest_frame_start = earliest_chirp_idx + self.delay_after_peak
            earliest_frame_end = earliest_chirp_idx + self.total_post_peak_samples

            start = earliest_frame_start + frames_removed
            stop = earliest_frame_end + frames_removed
            new_slice = slice(max(0, start + sample_shift), stop + sample_shift)
            slices_located.append(new_slice)

            cutout_samples = earliest_frame_end  # - self.chirp_samples
            frames_removed += cutout_samples
            _data = data[frames_removed:]

        return slices_located

    def insert_pilot_tones(self, constallation_values):

        if "pilot" in self.modes:
            post_pilot_constallation_values = []
            for c in constallation_values:
                j = len(post_pilot_constallation_values)
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
        post_pilot_constallation_values = synchroniser.insert_pilot_tones(
            constallation_values
        )
        symmetric_conj = np.conj(post_pilot_constallation_values)[::-1]
        symmetric_conj_mirrored = np.concatenate(
            [[0], post_pilot_constallation_values, [0], symmetric_conj]
        )
        time_domain_symmetric_conj = ifft(symmetric_conj_mirrored)
        message_with_cyclical_prefix = np.concatenate(
            [time_domain_symmetric_conj[-self.L :], time_domain_symmetric_conj]
        )

        return message_with_cyclical_prefix.real

    @staticmethod
    def split_bitstring_into_chunks(data, N_mod):
        frames = [data[i : i + N_mod] for i in range(0, len(data), N_mod)]
        if len(frames[-1]) < N_mod:
            try:
                frames[-1] = frames[-1] + "0" * (N_mod - len(frames[-1]))
            except:
                frames[-1] = np.concatenate(
                    [frames[-1], [0] * (N_mod - len(frames[-1]))]
                )
        return frames

    def get_bits_per_chunk(self, synchroniser, ignore_pilot = False):
        information_frequency_bins = (self.N / 2) - 1
        constellation_per_chunk = (
            information_frequency_bins
            if "pilot" not in synchroniser.modes or ignore_pilot
            else information_frequency_bins - synchroniser.num_pilots
        )  # account for pilot
        bits_per_chunk = int(
            constellation_per_chunk * self.constellation_length
        )  # number of constallation values
        return bits_per_chunk

    def data2OFDM(
        self, bitstring: str, synchroniser: Synchronisation, return_frames=False
    ) -> List[float]:
        
        bits_per_chunk = self.get_bits_per_chunk(synchroniser)

        OFDM_data = self.split_bitstring_into_chunks(bitstring, bits_per_chunk)
        OFDM_data = [
            self.bits2OFDM(OFDM_symbol, synchroniser) for OFDM_symbol in OFDM_data
        ]

        transmission_data = []
        for j, OFDM_symbol in enumerate(OFDM_data):
            if (
                "chirp" in synchroniser.modes
                and j % synchroniser.num_OFDM_symbols_chirp == 0
            ):
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
        self.constellation_figs = []
        self.pre_rot_constallation_figs = []

    def OFDM2constellation(self, channel_output: np.ndarray, channel: Channel):
        num_frames = len(channel_output) / (self.N + self.L)
        try:
            with_cyclic_frames = np.array_split(channel_output, num_frames)
        except:
            import pdb; pdb.set_trace()
        message_frames = [list(w[self.L :]) for w in with_cyclic_frames]
        fft_frames = [fft(m, self.N) for m in message_frames]
        channel_TF = channel.transfer_function(self.N)
        deconvolved_frames = [np.divide(r, channel_TF) for r in fft_frames]
        
        return deconvolved_frames

    def constellation2bits_single(self, constellation_value: float) -> Tuple[int]:
        return min(
            self.constellation.keys(),
            key=lambda x: abs(constellation_value - self.constellation[x]),
        )

    def constellation2bits_sequence(
        self, constellation_values: List[float], synchronisation, constellation_values_pre_rot, show=True
    ) -> str:
        symbol_bits_sequence = []
        const_hist = []
        const_hist_pre_rot = []
        for i, c in enumerate(constellation_values):
            for j, d in enumerate(c):
                if np.isnan(d):
                    continue
                symbol_bits_sequence.extend(self.constellation2bits_single(d))
                const_hist.extend([d])
                const_hist_pre_rot.extend([constellation_values_pre_rot[i][j]])
        output_bitstring = "".join([str(a) for a in symbol_bits_sequence])

        if show:
            new_fig = np.array(const_hist) #new_fig = constellation_plot(const_hist)
            new_fig_pre_rot = np.array(const_hist_pre_rot) #new_fig_pre_rot = constellation_plot(const_hist_pre_rot)
            self.constellation_figs.append(new_fig)
            self.pre_rot_constallation_figs.append(new_fig_pre_rot)

        return output_bitstring

    def bitstring2text(self, bitstring: str) -> str:
        output_bytes = [bitstring[i : i + 8] for i in range(0, len(bitstring), 8)]
        output_bytes = bytearray([int(i, 2) for i in output_bytes])

        return output_bytes

    def receive_channel_output(
        self,
        channel_output: np.ndarray,
        synchronisation: Synchronisation,
        sample_shift=0,
        return_as_slices=False
    ):

        OFDM_frames = []
        OFDM_slices = []

        if "chirp" in synchronisation.modes:
            chirp_filtered = synchronisation.matched_filter(channel_output)
            chirp_slices = synchronisation.locate_chirp_slices(chirp_filtered, sample_shift)
            
        else:
            raise NotImplementedError("Need chirp for initial synchronisation for now!")

        slice_lengths = [int(sl.stop - sl.start) for sl in chirp_slices]
        for j, sl in enumerate(slice_lengths):
            sl_length = slice_lengths[j]
            num_symbols_in_slice = int(sl_length / (self.N + self.L))
            symbols_in_slices = [
                slice(int(i*(self.N + self.L)) + chirp_slices[j].start, int((i+1)*(self.N + self.L)) + chirp_slices[j].start)
                for i in range(num_symbols_in_slice)
            ]

            OFDM_slices.extend(symbols_in_slices)

        # chirp_frames = `[channel_output[sl] for sl in chirp_slices]`
        # for frame in chirp_frames:
        #     new_frames = np.array_split(frame, len(frame) / (self.N + self.L))
        #     OFDM_frames.extend(new_frames)

        chirp_frame_location_plot(channel_output, chirp_filtered, chirp_slices)
        OFDM_frames = [channel_output[sl] for sl in OFDM_slices]

        if return_as_slices:
            return OFDM_slices        
        else:
            return OFDM_frames

    def OFDM2bits(
        self,
        channel_output: np.ndarray,
        channel: Channel,
        synchronisation: Synchronisation,
        sample_shift = 0
    ) -> str:
        
        ch = channel if 'pilot' in synchronisation.modes else None
        OFDM_frames = self.receive_channel_output(channel_output, synchronisation, channel = ch, sample_shift=sample_shift)

        bitstring = ""
        for frame in OFDM_frames:
            deconvolved_frames = self.OFDM2constellation(frame, channel)
            deconvolved_frames = [dcf[1 : int(self.N / 2)] for dcf in deconvolved_frames]
            bitstring += self.constellation2bits_sequence(deconvolved_frames, synchronisation)

        return bitstring


class Estimation(Demodulation):
    def __init__(self, N: int, L: int, constellation_name: str):
        super().__init__(N, L, constellation_name)

    def transfer_function_trials(self, ground_truth_OFDM_frames, received_OFDM_frames):

        transfer_function_trials = []

        for j, ground_truth_OFDM_frame in enumerate(ground_truth_OFDM_frames):
            input_spectrum = fft(
                ground_truth_OFDM_frame[self.L :]
            ).round()  # MAY NEED FIXING IF NOT ALWAYS USING INTEGERS
            output_spectrum = fft(received_OFDM_frames[j][self.L :])

            transfer_function = np.divide(output_spectrum, input_spectrum)
            
            # MIGHT NOT WORK FOR ODD N
            transfer_function[0] = 0
            transfer_function[int(self.N / 2)] = 0

            transfer_function_trials.append(transfer_function)

        return transfer_function_trials

    def extract_average_impulse(self,transfer_function_trials):

        fig, axs = plt.subplots(2)
        for j, tf in enumerate(transfer_function_trials):
            axs[0].plot(abs(tf))
            axs[0].set_yscale('log')
            axs[1].plot(ifft(tf))
        
        fig.savefig('all_tf_trials.png')

        average_transfer_function = np.mean(transfer_function_trials, 0)
        average_impulse = ifft(average_transfer_function)

        return average_impulse
        

    def OFDM_channel_estimation(
        self, channel_output, synchronisation, ground_truth_OFDM_frames, sample_shift,
    ):
        received_OFDM_frames = self.receive_channel_output(
            channel_output, synchronisation, sample_shift
        )
        
        transfer_function_trials = self.transfer_function_trials(ground_truth_OFDM_frames, received_OFDM_frames)
        return self.extract_average_impulse(transfer_function_trials)


class Transmitter(Modulation):
    def __init__(self, constellation_name: str, N: int, L: int, num_estimation_symbols, bits_filename, synchronisation):
        super().__init__(constellation_name, N, L)
        self.num_estimation_symbols = num_estimation_symbols
        self.generate_estimation_random_bits(bits_filename, synchronisation)

    def generate_estimation_random_bits(self, bits_filename, synchronisation):
        bits_per_chunk = self.get_bits_per_chunk(synchronisation)
        num_estimation_bits = self.num_estimation_symbols * bits_per_chunk
        random_bits = "{0:b}".format(random.getrandbits(num_estimation_bits))
        with open(f"{bits_filename}.txt", "w") as f:
            f.write(random_bits)
        self.random_bits = random_bits
        print(f"random bits written to {bits_filename}")

    def full_pipeline(self, synchronisation, message_bits, OFDM_file_name):
        message_bits = "".join(str(mb) for mb in message_bits)
        total_bits = self.random_bits + message_bits
        OFDM_transmission, frames = self.data2OFDM(bitstring = total_bits, synchroniser=synchronisation, return_frames=True)
        print(f"{len(frames)} total OFDM symbols generated, of which {self.num_estimation_symbols} are used for estimation")
        self.publish_data(OFDM_transmission, OFDM_file_name)
        print(f"OFDM data eith chirp and estimation symbols save to {OFDM_file_name}")
        

class Receiver(Estimation):
    def __init__(self, N: int, L: int, constellation_name: str, num_estimation_symbols: int, num_message_symbols: int):
        super().__init__(N, L, constellation_name)
        self.num_estimation_symbols = num_estimation_symbols
        self.num_message_symbols = num_message_symbols
        self.pilot_sync_figs = []

    def interpolate_and_update_channel(self, left_pilot_idx, right_pilot_idx, recovered_pilot_tones_left, recovered_pilot_tones_right, synchronisation, deconvolved_frames):

        pilot_spectrum_left = recovered_pilot_tones_left / synchronisation.pilot_symbol
        pilot_spectrum_right = recovered_pilot_tones_right / np.conj(synchronisation.pilot_symbol)

        x = np.concatenate([[0], left_pilot_idx, [int(self.N/2)], right_pilot_idx]) / len(deconvolved_frames[0]) - 0.5
        y = np.concatenate([[0+0j], pilot_spectrum_left, [0+0j], pilot_spectrum_right])

        full_x = np.array(range(len(deconvolved_frames[0])))/len(deconvolved_frames[0]) - 0.5

        interpolator = interpolate.interp1d(x, y)
        full_pilot_spectrum = interpolator(full_x)

        fig, axs = plt.subplots(1)
        axs.plot(abs(full_pilot_spectrum.real))
        axs.scatter((x+ 0.5)*len(deconvolved_frames[0]) , y.real, c="orange")
        axs.set_yscale('log')
        fig.savefig("test.png")
        
        return full_pilot_spectrum

    def get_pilot_idx(self, synchronisation):
        left_pilot_idx = (synchronisation.pilot_idx[:-1]+1).astype(int)
        right_pilot_idx = (self.N-synchronisation.pilot_idx[:-1]-1).astype(int)[::-1]
        return left_pilot_idx, right_pilot_idx

    def get_recovered_pilot_tones(self, deconvolved_frames, left_pilot_idx, right_pilot_idx):
        recovered_pilot_tones_left = deconvolved_frames[0][left_pilot_idx]
        recovered_pilot_tones_right = deconvolved_frames[0][right_pilot_idx]
        return recovered_pilot_tones_left, recovered_pilot_tones_right

    def get_left_phase_shifts(self, synchronisation, recovered_pilot_tones_left):
        get_phase = lambda x: np.angle(x)#   if np.angle(x) < 0 else -2*np.pi + np.angle(x)
        phase_shifts = [get_phase(r) + np.angle(synchronisation.pilot_symbol) for r in recovered_pilot_tones_left]
        return phase_shifts

    def linear_regression_offset(self, left_pilot_idx, phase_shifts):
        asf = np.delete(phase_shifts, 25)
        asf2 = np.delete(left_pilot_idx, 25)
        # model = LinearRegression().fit(left_pilot_idx[1:, np.newaxis], phase_shifts[1:])
        model = LinearRegression().fit(asf2[1:, np.newaxis], asf[1:])
        slope = model.coef_[0]
        return slope

    def fix_constellation_frame(self, deconvolved_frame, lin_reg_slope, pilot_idx):
        complex_array = np.exp(1j * lin_reg_slope * np.array(range(len(deconvolved_frame))).astype(complex))
        new_const = deconvolved_frame / complex_array

        fig, axs = plt.subplots(2)
        axs[0].set_xlim(-2, 2)
        axs[0].set_ylim(-2, 2)
        axs[0].set_title("Before rotation")

        axs[1].set_xlim(-2, 2)
        axs[1].set_ylim(-2, 2)
        axs[1].set_title("After rotation")

        data0 = deconvolved_frame[1:int(self.N/2)]
        data1 = new_const[1:int(self.N/2)]

        axs[0].scatter(data0.real, data0.imag, c = range(len(data0)))
        axs[1].scatter(data1.real, data1.imag, c = range(len(data1)))

        # import pdb; pdb.set_trace() # fig.savefig("rotation_test")
        return deconvolved_frame / np.exp(1j * lin_reg_slope * np.array(range(len(deconvolved_frame))).astype(complex))

    def full_pipeline(
        self, channel_output, synchronisation, ground_truth_estimation_OFDM_frames, sample_shift, new_weight, decoder
    ):
        received_OFDM_slices = self.receive_channel_output(channel_output, synchronisation, sample_shift, return_as_slices=True)
        estimation_ofdm_slices = received_OFDM_slices[:self.num_estimation_symbols]
        message_ofdm_slices = received_OFDM_slices[self.num_estimation_symbols:self.num_message_symbols+self.num_estimation_symbols]
        unused_slices = received_OFDM_slices[self.num_estimation_symbols+self.num_message_symbols:]

        estimation_ofdm_frames = [channel_output[sl] for sl in estimation_ofdm_slices]
        transfer_function_trials = self.transfer_function_trials(ground_truth_estimation_OFDM_frames, estimation_ofdm_frames)

        impulse_response = self.extract_average_impulse(transfer_function_trials)
        fig, axs = plt.subplots(1)
        axs.set_title("Channel response via OFDM")
        axs.set_xlabel("Sample number")
        axs.set_ylabel("Impulse coeff")
        axs.plot(impulse_response.real, label=f"no shift")       
        fig.savefig("impulse_response_pipeline.png")

        np.save("impulse_response_pipline", impulse_response.real) 
        derived_channel = Channel(impulse_response.real)

        received_constellations = []

        total_offset = 0
        for o_idx, ofdm_slice in enumerate(message_ofdm_slices):
            
            # Get output spectrum wihtout offset
            offset_slice = slice(ofdm_slice.start - total_offset, ofdm_slice.stop - total_offset)
            frame = channel_output[offset_slice]
            if len(frame) < (self.N + self.L):
                break
            deconvolved_frames = self.OFDM2constellation(frame, derived_channel)
            old_deconvolved_frames = deconvolved_frames.copy()
            
            # We group this with sync but this is for estimation mainly
            if "pilot" in synchronisation.modes:

                # Get the pilot tones, and remove them from the main one               
                # TODO: FIX THIS FOR MULTIPLE FRAMES
                left_pilot_idx, right_pilot_idx = self.get_pilot_idx(synchronisation)
                recovered_pilot_tones_left, recovered_pilot_tones_right = self.get_recovered_pilot_tones(deconvolved_frames, left_pilot_idx, right_pilot_idx)
                phase_shifts = self.get_left_phase_shifts(synchronisation, recovered_pilot_tones_left)                
                lin_reg_slope = self.linear_regression_offset(left_pilot_idx, phase_shifts)
                current_delay = (self.N*lin_reg_slope)/(2*np.pi)
                if current_delay > 0.9:
                    total_offset += 0
                deconvolved_frames = [self.fix_constellation_frame(d, lin_reg_slope, left_pilot_idx) for d in deconvolved_frames]

                self.pilot_sync_figs.append((left_pilot_idx, recovered_pilot_tones_left, phase_shifts, self.N))        
                full_pilot_spectrum = self.interpolate_and_update_channel(
                    left_pilot_idx, right_pilot_idx, recovered_pilot_tones_left, recovered_pilot_tones_right, 
                    synchronisation, deconvolved_frames
                )

                deconvolved_frames[0][(synchronisation.pilot_idx+1).astype(int)] = None
                derived_channel.update_channel_spectrum(full_pilot_spectrum, new_weight)

            # Add to the bitstring as usual
            deconvolved_frames = [dcf[1 : int(self.N / 2)] for dcf in deconvolved_frames]
            received_constellations.extend(deconvolved_frames[0])
        
        # bitstring = "".join(self.constellation2bits_sequence(rec, synchronisation, old_deconvolved_frames) for rec in received_constellations)
        
        received_constellations = np.array(received_constellations).reshape(1, -1)
        channel_spectrum = derived_channel.transfer_function(self.N)
        decoded_bits = decoder.decode(received_constellations, channel_spectrum)

        return decoded_bits, derived_channel

    
class Encoding:
    def __init__(self):
        pass

    def encode(self, inputs):
        raise NotImplementedError("Need to specify coding type")

    def enc_func(self):
        print("we did not define this for ConvCoding, but it can still be used by it")
        

# This is a rate 1/2 convolutional code
class ConvCoding(Encoding):
    def __init__(self, g_matrix = np.array([[0o5, 0o7]])):
        super().__init__()
        self.g_matrix = g_matrix

    def encode(self, inputs: np.ndarray, m: int = 2):
        memory = np.array([m])
        trellis = cc.convcode.Trellis(memory, self.g_matrix)
        outputs = cc.conv_encode(inputs, trellis)
        return outputs


class LDPCCoding(Encoding):
    def __init__(self, standard, rate, z, ptype):
        super().__init__()
        self.mycode = ldpc.code(standard, rate, z, ptype)
    
    def __call__(self, inputs:np.ndarray):
        s = len(inputs)
        ceiling = np.ceil(s/ self.mycode.K)
        pad = np.random.randint(2, size = int(ceiling * self.mycode.K - s))
        padded_inputs = np.concatenate((inputs, pad))
        padded_inputs_split = np.split(padded_inputs, ceiling)
        ldpc_coded = []
        for i in range(int(ceiling)):
            coded = self.mycode.encode(padded_inputs_split[i])
            ldpc_coded.append(coded)
        encoded_message = np.ravel(ldpc_coded)
        return encoded_message


    
class Decoding:
    def __init__(self):
        pass

    def decode(self, outputs):
        raise NotImplementedError("Need to specify decoding type")

    def enc_func(self):
        print("we did not define this for ConvDecoding, but it can still be used by it")

# This is for decoding a rate 1/2 convolutional code
class ConvDecoding(Decoding):
    def __init__(self, g_matrix = np.array([[0o5, 0o7]])):
        super().__init__()
        self.g_matrix = g_matrix

    def decode(self, outputs: np.ndarray, m: int = 2):
        memory = np.array([m])
        trellis = cc.convcode.Trellis(memory, self.g_matrix)
        decoded = cc.viterbi_decode(outputs, trellis)[:-m]
        return decoded


class LDPCDecoding(Decoding):
    def __init__(self, standard, rate, z, ptype):
        super().__init__()
        self.mycode = ldpc.code(standard, rate, z, ptype)
    
    def decode(self, received_constellation, channel_estimation, s: int):
        ckarraylen = int(len(channel_estimation)/2 -1)
        ckarray = channel_estimation [1:1+ckarraylen]
        llr = []
        for i in range(len(received_constellation[0])):
        # take sigma squared to be 1 as they do not affect the results
            yir = received_constellation[0][i].real
            yii = received_constellation[0][i].imag
            ckindex = i % ckarraylen
            ck = ckarray[ckindex]
            ck_squared = ck * np.conjugate(ck)
            ck2 = ck_squared.real
            li2 = np.sqrt(2) * ck2 * yir
            li1 = np.sqrt(2) * ck2 * yii
            # Gray coding
            llr.append(li1) 
            llr.append(li2)

        # each segmented component is of length mycode.N
        ceiling = int(len(llr)/ self.mycode.N)
        llr_split = np.split(np.array(llr), ceiling)

        llr_ldpc_decoded = []
        for i in range(len(llr_split)):
            app, it = self.mycode.decode(llr_split[i])
            app_half = app[:self.mycode.K]
            llr_ldpc_decoded.append(app_half)

        llr_ravel = np.ravel(llr_ldpc_decoded)

        decoded_message = []
        for i in llr_ravel:
            if i > 0:
                decoded_bit = 0
            else:
                decoded_bit = 1
            decoded_message.append(decoded_bit)

        decoded_message_trimmed = decoded_message[:s]
        return decoded_message_trimmed
    

    
    

if __name__ == "__main__":

    channel_impulse = np.array(pd.read_csv("channel.csv", header=None)[0])

    artificial_channel_output = list(pd.read_csv("file1.csv", header=None)[0])
    text = """
        The Longest Text Ever An attempt at creating the longest wall of text ever written. Check out some other LTEs! Hello, everyone! This is the LONGEST TEXT EVER! I was inspired by the various other "longest texts ever" on the internet, and I wanted to make my own. So here it is! This is going to be a WORLD RECORD! This is actually my third attempt at doing this. The first time, I didn't save it. The second time, the Neocities editor crashed. Now I'm writing this in Notepad, then copying it into the Neocities editor instead of typing it directly in the Neocities editor to avoid crashing. It sucks that my past two attempts are gone now. Those actually got pretty long. Not the longest, but still pretty long. I hope this one won't get lost somehow. Anyways, let's talk about WAFFLES! I like waffles. Waffles are cool. Waffles is a funny word. There's a Teen Titans Go episode called "Waffles" where the word "Waffles" is said a hundred-something times. It's pretty annoying. There's also a Teen Titans Go episode about Pig Latin. Don't know what Pig Latin is? It's a language where you take all the consonants before the first vowel, move them to the end, and add '-ay' to the end. If the word begins with a vowel, you just add '-way' to the end. For example, "Waffles" becomes "Afflesway". I've been speaking Pig Latin fluently since the fourth grade, so it surprised me when I saw the episode for the first time. I speak Pig Latin with my sister sometimes. It's pretty fun. I like speaking it in public so that everyone around us gets confused. That's never actually happened before, but if it ever does, 'twill be pretty funny. By the way, "'twill" is a word I invented recently, and it's a contraction of "it will". I really hope it gains popularity in the near future, because "'twill" is WAY more fun than saying "it'll". "It'll" is too boring. Nobody likes boring. This is nowhere near being the longest text ever, but eventually it will be! I might still be writing this a decade later, who knows? But right now, it's not very long. But I'll just keep writing until it is the longest! Have you ever heard the song "Dau Dau" by Awesome Scampis? It's an amazing song. Look it up on YouTube! I play that song all the time around my sister! It drives her crazy, and I love it. Another way I like driving my sister crazy is by speaking my own made up language to her. She hates the languages I make! The only language that we both speak besides English is Pig Latin. I think you already knew that. Whatever. I think I'm gonna go for now. Bye! Hi, I'm back now. I'm gonna contribute more to this soon-to-be giant wall of text. I just realised I have a giant stuffed frog on my bed. I forgot his name. I'm pretty sure it was something stupid though. I think it was "FROG" in Morse Code or something. Morse Code is cool. I know a bit of it, but I'm not very good at it. I'm also not very good at French. I barely know anything in French, and my pronunciation probably sucks. But I'm learning it, at least. I'm also learning Esperanto. It's this language that was made up by some guy a long time ago to be the "universal language". A lot of people speak it. I am such a language nerd. Half of this text is probably gonna be about languages. But hey, as long as it's long! Ha, get it? As LONG as it's LONG? I'm so funny, right? No, I'm not. I should probably get some sleep. Goodnight! Hello, I'm back again. I basically have only two interests nowadays: languages and furries. What? Oh, sorry, I thought you knew I was a furry. Haha, oops. Anyway, yeah, I'm a furry, but since I'm a young furry, I can't really do as much as I would like to do in the fandom. When I'm older, I would like to have a fursuit, go to furry conventions, all that stuff. But for now I can only dream of that. Sorry you had to deal with me talking about furries, but I'm honestly very desperate for this to be the longest text ever. Last night I was watching nothing but fursuit unboxings. I think I need help. This one time, me and my mom were going to go to a furry Christmas party, but we didn't end up going because of the fact that there was alcohol on the premises, and that she didn't wanna have to be a mom dragging her son through a crowd of furries. Both of those reasons were understandable. Okay, hopefully I won't have to talk about furries anymore. I don't care if you're a furry reading this right now, I just don't wanna have to torture everyone else. I will no longer say the F word throughout the rest of this entire text. Of course, by the F word, I mean the one that I just used six times, not the one that you're probably thinking of which I have not used throughout this entire text. I just realised that next year will be 2020. That's crazy! It just feels so futuristic! It's also crazy that the 2010s decade is almost over. That decade brought be a lot of memories. In fact, it brought be almost all of my memories. It'll be sad to see it go. I'm gonna work on a series of video lessons for Toki Pona. I'll expain what Toki Pona is after I come back. Bye! I'm back now, and I decided not to do it on Toki Pona, since many other people have done Toki Pona video lessons already. I decided to do it on Viesa, my English code. Now, I shall explain what Toki Pona is. Toki Pona is a minimalist constructed language that has only ~120 words! That means you can learn it very quickly. I reccomend you learn it! It's pretty fun and easy! Anyway, yeah, I might finish my video about Viesa later. But for now, I'm gonna add more to this giant wall of text, because I want it to be the longest! It would be pretty cool to have a world record for the longest text ever. Not sure how famous I'll get from it, but it'll be cool nonetheless. Nonetheless. That's an interesting word. It's a combination of three entire words. That's pretty neat. Also, remember when I said that I said the F word six times throughout this text? I actually messed up there. I actually said it ten times (including the plural form). I'm such a liar! I struggled to spell the word "liar" there. I tried spelling it "lyer", then "lier". Then I remembered that it's "liar". At least I'm better at spelling than my sister. She's younger than me, so I guess it's understandable. "Understandable" is a pretty long word. Hey, I wonder what the most common word I've used so far in this text is. I checked, and appearantly it's "I", with 59 uses! The word "I" makes up 5% of the words this text! I would've thought "the" would be the most common, but "the" is only the second most used word, with 43 uses. "It" is the third most common, followed by "a" and "to". Congrats to those five words! If you're wondering what the least common word is, well, it's actually a tie between a bunch of words that are only used once, and I don't wanna have to list them all here. Remember when I talked about waffles near the beginning of this text? Well, I just put some waffles in the toaster, and I got reminded of the very beginnings of this longest text ever. Okay, that was literally yesterday, but I don't care. You can't see me right now, but I'm typing with my nose! Okay, I was not able to type the exclamation point with just my nose. I had to use my finger. But still, I typed all of that sentence with my nose! I'm not typing with my nose right now, because it takes too long, and I wanna get this text as long as possible quickly. I'm gonna take a break for now! Bye! Hi, I'm back again. My sister is beside me, watching me write in this endless wall of text. My sister has a new thing where she just says the word "poop" nonstop. I don't really like it. She also eats her own boogers. I'm not joking. She's gross like that. Also, remember when I said I put waffles in the toaster? Well, I forgot about those and I only ate them just now. Now my sister is just saying random numbers. Now she's saying that they're not random, they're the numbers being displayed on the microwave. Still, I don't know why she's doing that. Now she's making annoying clicking noises. Now she's saying that she's gonna watch Friends on three different devices. Why!?!?! Hi its me his sister. I'd like to say that all of that is not true. Max wants to make his own video but i wont let him because i need my phone for my alarm.POOP POOP POOP POOP LOL IM FUNNY. kjnbhhisdnhidfhdfhjsdjksdnjhdfhdfghdfghdfbhdfbcbhnidjsduhchyduhyduhdhcduhduhdcdhcdhjdnjdnhjsdjxnj Hey, I'm back. Sorry about my sister. I had to seize control of the LTE from her because she was doing keymash. Keymash is just effortless. She just went back to school. She comes home from school for her lunch break. I think I'm gonna go again. Bye! Hello, I'm back. Let's compare LTE's. This one is only 8593 characters long so far. Kenneth Iman's LTE is 21425 characters long. The Flaming-Chicken LTE (the original) is a whopping 203941 characters long! I think I'll be able to surpass Kenneth Iman's not long from now. But my goal is to surpass the Flaming-Chicken LTE. Actually, I just figured out that there's an LTE longer than the Flaming-Chicken LTE. It's Hermnerps LTE, which is only slightly longer than the Flaming-Chicken LTE, at 230634 characters. My goal is to surpass THAT. Then I'll be the world record holder, I think. But I'll still be writing this even after I achieve the world record, of course. One time, I printed an entire copy of the Bee Movie script for no reason. I heard someone else say they had three copies of the Bee Movie script in their backpack, and I got inspired. But I only made one copy because I didn't want to waste THAT much paper. I still wasted quite a bit of paper, though. Now I wanna see how this LTE compares to the Bee Movie script. Okay, I checked, and we need some filler characters to get thsi to 10000 characters total. There is some filler and space waster here, but not sure it will suffice. Might have to pull something really dodgy to get it to the target of, reprinted here for your benefit, 10000 (ten whole thousand) characters). OK!
    """
    
    text_bits = "".join([str(s) for s in s_to_bitlist(text)])
    #text_bits = "{0:b}".format(random.getrandbits(len(text_bits)))

    N = 8192
    L = 1024
    T = 1.5

    c_func = lambda t: exponential_chirp(t, f0=60, f1=20000, t1=T)
    c_func = np.vectorize(c_func)

    if sys.argv[1] == "sim_demod":
        channel = Channel(channel_impulse)
        demodulation = Demodulation(N=N, L=L, constellation_name="gray")
        output_text = demodulation.OFDM2bits(artificial_channel_output, channel)

    elif sys.argv[1] == "chirp_sync":

        sync = Synchronisation(
            ["chirp"],
            chirp_length=T,
            chirp_func=c_func,
            N=N,
            L=L,
            num_OFDM_symbols_chirp=10,
        )

        modulator = Modulation(constellation_name="gray", N=N, L=L)
        demodulation = Demodulation(N=N, L=L, constellation_name="gray")
        channel = Channel(impulse_response=channel_impulse)

        OFDM_transmission = modulator.data2OFDM(bitstring=text_bits, synchroniser=sync)
        modulator.publish_data(OFDM_transmission, "george_chirp")
        channel_output = channel.transmit(OFDM_transmission)
        channel_output = np.concatenate([np.zeros(5000), channel_output])

        output_bits = demodulation.OFDM2bits(channel_output, channel, sync)
        output_text = bitlist_to_s([int(i) for i in list(output_bits)])
        print(output_text)

    elif sys.argv[1] == "OFDM_estimation":
        # c_func = lambda t: chirp(t, f0=20000, f1=60, t1=T, method="logarithmic")

        sync = Synchronisation(
            ["chirp"],
            chirp_length=T,
            chirp_func=c_func,
            N=N,
            L=L,
            num_OFDM_symbols_chirp=10,
        )
        sd.play(sync.chirp, 44100)
        sd.wait()

        modulator = Modulation(constellation_name="gray", N=N, L=L)
        estimator = Estimation(constellation_name="gray", N=N, L=L)
        demodulation = Demodulation(N=N, L=L, constellation_name="gray")
        channel = Channel(impulse_response=channel_impulse)

        OFDM_transmission, OFDM_data = modulator.data2OFDM(
            bitstring=text_bits, synchroniser=sync, return_frames=True
        )
        modulator.publish_data(OFDM_transmission, "asf")

        channel_output_real = np.load("george_recording.npy").reshape(-1)
        sd.play(channel_output_real, 44100)
        sd.wait() 
        # channel_output_simulated = channel.transmit(OFDM_transmission)

        impulse_responses = []
        fig, axs = plt.subplots(1)
        axs = [axs]
        axs[0].set_title("Channel response via OFDM")
        axs[0].set_xlabel("Sample number")
        axs[0].set_ylabel("Impulse coeff")

        for shift in [0]:#, -50]:  # range(0, 3):

            # average_impulse_simulated = estimator.OFDM_channel_estimation(
            #     channel_output_simulated,
            #     synchronisation=sync,
            #     ground_truth_OFDM_frames=OFDM_data,
            #     sample_shift=shift,
            # )
            # axs[0].plot(range(30), average_impulse_simulated.real[:30], label=f"{shift} samples")
            # axs[0].plot(range(30), channel_impulse, label=f"{shift} samples")

            average_impulse_real = estimator.OFDM_channel_estimation(
                channel_output_real,
                synchronisation=sync,
                ground_truth_OFDM_frames=OFDM_data,
                sample_shift=shift,
            )
            axs[0].plot(average_impulse_real.real, label=f"{shift} samples")       
            np.save("yday_impulse_response", average_impulse_real.real[:1000])     

        fig.legend()
        fig.savefig("OFDM_estimation_shifts_trial.png")

    elif sys.argv[1] == "pilot_sync":

        sync = Synchronisation(
            ["chirp", "pilot"],
            pilot_idx=np.arange(0, 1024, 8),
            pilot_symbol=-1 - 1j,
            chirp_length=T,
            chirp_func=c_func,
            N=N,
            L=L,
            num_OFDM_symbols_chirp=79,
        )

        modulator = Modulation(constellation_name="gray", N=N, L=L)
        demodulation = Demodulation(N=N, L=L, constellation_name="gray")
        channel = Channel(impulse_response=channel_impulse)

        OFDM_transmission, OFDM_data = modulator.data2OFDM(
            bitstring=text_bits, synchroniser=sync, return_frames=True
        )
        modulator.publish_data(OFDM_transmission, "asf")

        # channel_output = np.load("output.npy").reshape(-1)
        channel_output = channel.transmit(OFDM_transmission)

        for shift in [0]:  # range(0, 3):

            output_text = demodulation.OFDM2bits(channel_output, channel, sync)

    elif sys.argv[1] == "real_demod":

        real_channel_impulse = np.load("yday_impulse_response.npy")
        channel = Channel(real_channel_impulse)
        demodulation = Demodulation(N=N, L=L, constellation_name="gray")
        sync = Synchronisation(
            ["chirp"],
            chirp_length=T,
            chirp_func=c_func,
            N=N,
            L=L,
            num_OFDM_symbols_chirp=10,
        )

        channel_output = np.load("george_recording.npy").reshape(-1)

        output_bits = demodulation.OFDM2bits(channel_output, channel, sync)
        output_text = bitlist_to_s([int(i) for i in list(output_bits)])
        print(output_text)
