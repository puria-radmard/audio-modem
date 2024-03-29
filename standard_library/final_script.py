from os import error

from numpy.testing._private.utils import print_assert_equal
from util_classes import Modulation, Transmitter, Receiver, Synchronisation, LDPCCoding, LDPCDecoding, Channel
import sys
from util_objects import *
import sounddevice as sd
from tempfile import TemporaryFile
import numpy as np
import pandas as pd
from testing_script import get_OFDM_data_from_bits
from visualisation_scripts import generate_constellation_video, generate_phaseshifting_video, generate_channel_estim_video
from matplotlib.cbook.deprecation import MatplotlibDeprecationWarning

import warnings
warnings.simplefilter('ignore', np.ComplexWarning)
warnings.simplefilter('ignore', RuntimeWarning)
warnings.simplefilter('ignore', MatplotlibDeprecationWarning)


fs=44100
recording_duration = 15 # seconds
N = int(2048)
L = 256
T = 2

num_estimation_symbol = 10

c_func = lambda t: exponential_chirp(t, f0=60, f1=16000, t1=T)
c_func = np.vectorize(c_func)

message = """\x94... An attempt at creating the longest wall of text ever written. Check out some other LTEs! Hello, everyone! This is the LONGEST TEXT EVER! I was inspired by the various other "longest texts ever" on the internet, and I wanted to make my own. So here it is! This is going to be a WORLD RECORD! This is actually my third attempt at doing this. The first time, I didn't save it. The second time, the Neocities editor crashed. Now I'm writing this in Notepad, then copying it into the Neocities editor instead of typing it directly in the Neocities editor to avoid crashing. It sucks that my past two attempts are gone now. Those actually got pretty long. Not the longest, but still pretty long. I hope this one won't get lost somehow. Anyways, let's talk about WAFFLES! I like waffles. Waffles are cool. Waffles is a funny word. There's a Teen Titans Go episode called "Waffles" where the word "Waffles" is said a hundred-something times. It's pretty annoying. There's also a Teen Titans Go episode about Pig Latin. Don't know what Pig Latin is? It's a language where you take all the consonants before the first vowel, move them to the end, and add '-ay' to the end. If the word begins with a vowel, you just add '-way' to the end. For example, "Waffles" becomes "Afflesway". I've been speaking Pig Latin fluently since the fourth grade, so it surprised me when I saw the episode for the first time. I speak Pig Latin with my sister sometimes. It's pretty fun. I like speaking it in public so that everyone around us gets confused. That's never actually happened before, but if it ever does, 'twill be pretty funny. By the way, "'twill" is a word I invented recently, and it's a contraction of "it will". I really hope it gains popularity in the near future, because "'twill" is WAY more fun than saying "it'll". "It'll" is too boring. Nobody likes boring. This is nowhere near being the longest text ever, but eventually it will be! I might still be writing this a decade later, who knows? But right now, it's not very long. But I'll just keep writing until it is the longest! Have you ever heard the song "Dau Dau" by Awesome Scampis? It's an amazing song. Look it up on YouTube! I play that song all the time around my sister! It drives her crazy, and I love it. Another way I like driving my sister crazy is by speaking my own made up language to her. She hates the languages I make! The only language that we both speak besides English is Pig Latin. I think you already knew that. Whatever. I think I'm gonna go for now. Bye! Hi, I'm back now. I'm gonna contribute more to this soon-to-be giant wall of text. I just realised I have a giant stuffed frog on my bed. I forgot his name. I'm pretty sure it was something stupid though. I think it was "FROG" in Morse Code or something. Morse Code is cool. I know a bit of it, but I'm not very good at it. I'm also not very good at French. I barely know anything in French, and my pronunciation probably sucks. But I'm learning it, at least. I'm also learning Esperanto. It's this language that was made up by some guy a long time ago to be the "universal language". A lot of people speak it. I am such a language nerd. Half of this text is probably gonna be about languages. But hey, as long as it's long! Ha, get it? As LONG as it's LONG? I'm so funny, right? No, I'm not. I should probably get some sleep. Goodnight! Hello, I'm back again. I basically have only two interests nowadays: languages and furries. What? Oh, sorry, I thought you knew I was a furry. Haha, oops. Anyway, yeah, I'm a furry, but since I'm a young furry, I can't really do as much as I would like to do in the fandom. When I'm older, I would like to have a fursuit, go to furry conventions, all that stuff. But for now I can only dream of that. Sorry you had to deal with me talking about furries, but I'm honestly very desperate for this to be the longest text ever. Last night I was watching nothing but fursuit unboxings. I think I need help. This one time, me and my mom were going to go to a furry Christmas party, but we didn't end up going because of the fact that there was alcohol on the premises, and that she didn't wanna have to be a mom dragging her son through a crowd of furries. Both of those reasons were understandable. Okay, hopefully I won't have to talk about furries anymore. I don't care if you're a furry reading this right now, I just don't wanna have to torture everyone else. I will no longer say the F word throughout the rest of this entire text. Of course, by the F word, I mean the one that I just used six times, not the one that you're probably thinking of which I have not used throughout this entire text. I just realised that next year will be 2020. That's crazy! It just feels so futuristic! It's also crazy that the 2010s decade is almost over. That decade brought be a lot of memories. In fact, it brought be almost all of my memories. It'll be sad to see it go. I'm gonna work on a series of video lessons for Toki Pona. I'll expain what Toki Pona is after I come back. Bye! I'm back now, and I decided not to do it on Toki Pona, since many other people have done Toki Pona video lessons already. I decided to do it on Viesa, my English code. Now, I shall explain what Toki Pona is. Toki Pona is a minimalist constructed language that has only ~120 words! That means you can learn it very quickly. I reccomend you learn it! It's pretty fun and easy! Anyway, yeah, I might finish my video about Viesa later. But for now, I'm gonna add more to this giant wall of text, because I want it to be the longest! It would be pretty cool to have a world record for the longest text ever. Not sure how famous I'll get from it, but it'll be cool nonetheless. Nonetheless. That's an interesting word. It's a combination of three entire words. That's pretty neat. Also, remember when I said that I said the F word six times throughout this text? I actually messed up there. I actually said it ten times (including the plural form). I'm such a liar! I struggled to spell the word "liar" there. I tried spelling it "lyer", then "lier". Then I remembered that it's "liar". At least I'm better at spelling than my sister. She's younger than me, so I guess it's understandable. "Understandable" is a pretty long word. Hey, I wonder what the most common word I've used so far in this text is. I checked, and appearantly it's "I", with 59 uses! The word "I" makes up 5% of the words this text! I would've thought "the" would be the most common, but "the" is only the second most used word, with 43 uses. "It" is the third most common, followed by "a" and "to". Congrats to those five words! If you're wondering what the least common word is, well, it's actually a tie between a bunch of words that are only used once, and I don't wanna have to list them all here. Remember when I talked about waffles near the beginning of this text? Well, I just put some waffles in the toaster, and I got reminded of the very beginnings of this longest text ever. Okay, that was literally yesterday, but I don't care. You can't see me right now, but I'm typing with my nose! Okay, I was not able to type the exclamation point with just my nose. I had to use my finger. But still, I typed all of that sentence with my nose! I'm not typing with my nose right now, because it takes too long, and I wanna get this text as long as possible quickly. I'm gonna take a break for now! Bye! Hi, I'm back again. My sister is beside me, watching me write in this endless wall of text. My sister has a new thing where she just says the word "poop" nonstop. I don't really like it. She also eats her own boogers. I'm not joking. She's gross like that. Also, remember when I said I put waffles in the toaster? Well, I forgot about those and I only ate them just now. Now my sister is just saying random numbers. Now she's saying that they're not random, they're the numbers being displayed on the microwave. Still, I don't know why she's doing that. Now she's making annoying clicking noises. Now she's saying that she's gonna watch Friends on three different devices. Why!?!?! Hi its me his sister. I'd like to say that all of that is not true. Max wants to make his own video but i wont let him because i need my phone for my alarm.POOP POOP POOP POOP LOL IM FUNNY. kjnbhhisdnhidfhdfhjsdjksdnjhdfhdfghdfghdfbhdfbcbhnidjsduhchyduhyduhdhcduhduhdcdhcdhjdnjdnhjsdjxnj Hey, I'm back. Sorry about my sister. I had to seize control of the LTE from her because she was doing keymash. Keymash is just effortless. She just went back to school. She comes home from school for her lunch break. I think I'm gonna go again. Bye! Hello, I'm back. Let's compare LTE's. This one is only 8593 characters long so far. Kenneth Iman's LTE is 21425 characters long. The Flaming-Chicken LTE (the original) is a whopping 203941 characters long! I think I'll be able to surpass Kenneth Iman's not long from now. But my goal is to surpass the Flaming-Chicken LTE. Actually, I just figured out that there's an LTE longer than the Flaming-Chicken LTE. It's Hermnerps LTE, which is only slightly longer than the Flaming-Chicken LTE, at 230634 characters. My goal is to surpass THAT. Then I'll be the world record holder, I think. But I'll still be writing this even after I achieve the world record, of course. One time, I printed an entire copy of the Bee Movie script for no reason. I heard someone else say they had three copies of the Bee Movie script in their backpack, and I got inspired. But I only made one copy because I didn't want to waste THAT much paper. I still wasted quite a bit of paper, though. Now I wanna see how this LTE compares to the Bee Movie script. Okay, I checked, and we need some filler characters to get thsi to 10000 characters total. There is some filler and space waster here, but not sure it will suffice. Might have to pull something really dodgy to get it to the target of, reprinted here for your benefit, 10000 (ten whole thousand) characters). OK!"""
uncoded_text_bits = "".join([str(s) for s in s_to_bitlist(message)])
with open("unencoded_bits", "w") as f:
    f.write("".join(str(s) for s in uncoded_text_bits))
l_before = len(uncoded_text_bits)

sync = Synchronisation(["chirp", "pilot"], pilot_idx=np.arange(0, N/2, 32), pilot_symbol=-1 - 1j,chirp_length=T, chirp_func=c_func, N=N, L=L, num_OFDM_symbols_chirp=100)
# sync = Synchronisation(["chirp"], chirp_length=T, chirp_func=c_func, N=N, L=L, num_OFDM_symbols_chirp=71)

encoder = LDPCCoding(standard = '802.11n', rate = '1/2', z=27, ptype='A')
decoder = LDPCDecoding(standard = '802.11n', rate = '1/2', z=27, ptype='A')
# encoded_text_bits = uncoded_text_bits 
encoded_text_bits = encoder(np.array([int(b) for b in uncoded_text_bits]))
with open("encoded_bits", "w") as f:
    f.write("".join(str(s) for s in encoded_text_bits))

print(f"Length of bits before encoding = {l_before}")
print(f"Length of bits after encoding = {len(encoded_text_bits)}")
print(f"Rate = {l_before/len(encoded_text_bits)}")

try:
    reader_idx = sys.argv[-1]
except:
    reader_idx = ""


if sys.argv[1] == 'send':
    transmitter = Transmitter("gray", N=N, L=L, num_estimation_symbols=num_estimation_symbol, bits_filename=f"full_pipeline_random_bits{reader_idx}", synchronisation=sync)
    transmitter.full_pipeline(sync, encoded_text_bits, f"full_pipeline_transmission_audio{reader_idx}")

elif sys.argv[1] == 'receive':

    receiver = Receiver(N=N, L=L, constellation_name="gray", num_estimation_symbols=num_estimation_symbol, num_message_symbols=100)
    recording_file = f"final_pipeline_message_with_estimation{reader_idx}"
    video_folder_name = "videos_final"

    # outfile = TemporaryFile()
    # print("recording started!")
    # myrecording = sd.rec(int(recording_duration * fs), samplerate=fs, channels=2)
    # sd.wait()
    # myrecording = np.array(myrecording)
    # myrecording = np.mean(myrecording, 1)
    # np.save(recording_file, myrecording)
    # print("recording done!")
    channel_output = np.load(f"{recording_file}.npy").reshape(-1)

    # artificial_channel_impulse = np.array(pd.read_csv("channel.csv", header=None)[0])
    # channel = Channel(artificial_channel_impulse)
    # channel_input = np.load(f"full_pipeline_transmission_audio{reader_idx}.npy").reshape(-1)
    # channel_output = channel.transmit(channel_input, noise = 0.01)

    modu = Modulation("gray", N, L)

    ground_truth_estimation_OFDM_frames = get_OFDM_data_from_bits(modu, sync, source_bits = f"full_pipeline_random_bits{reader_idx}")
    for nw in [0]:#, 0.1, 0.15, 0.2, 0.25]:
        received_bitstring, inferred_channel = receiver.full_pipeline(
            channel_output, sync, ground_truth_estimation_OFDM_frames, sample_shift = 0, new_weight = nw, decoder = decoder
        )
        #received_bitstring = np.array(received_bitstring)
        received_bitstring = "".join(str(r) for r in received_bitstring)
        import pdb; pdb.set_trace()
        
        ### TODO: UNDERSTAND THIS
        # received_bitstring = '1' + received_bitstring
        error_rate = 1-sum(uncoded_text_bits[i] == received_bitstring[i] for i in range(len(uncoded_text_bits)))/len(uncoded_text_bits)        

        print(nw, error_rate)
        if not nw:
            pass
            #generate_constellation_video(video_folder_name, receiver.constellation_figs, receiver.pre_rot_constallation_figs, f"constalletion_withpilotsync_nw{nw}")
        #generate_phaseshifting_video(video_folder_name, receiver.pilot_sync_figs, f"pilotphaseshift_withpilotsync_nw{nw}", sync.pilot_symbol)        
        #generate_channel_estim_video(video_folder_name, inferred_channel, f"channelupdates_nopilotsync_nw{nw}")

    output_text = bitlist_to_s([int(r) for r in list(received_bitstring[:len(uncoded_text_bits)])])
    print(output_text)
