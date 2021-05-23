import sys
import numpy as np
from util_classes import Estimation, Modulation, Synchronisation, Demodulation, Channel
import random
import sounddevice as sd
from tempfile import TemporaryFile
from util_objects import *
import matplotlib.pyplot as plt

def publish_random_bits(file_name, binary_length):
    text_bits = "{0:b}".format(random.getrandbits(binary_length))
    with open(f"{file_name}.txt", "w") as f:
        f.write(text_bits)
    print(f"random bits written to {file_name}")


def get_OFDM_data_from_bits(modulator, sync, source_bits):
    with open(f"{source_bits}.txt") as f:
        text_bits = f.read()
    OFDM_transmission, OFDM_data = modulator.data2OFDM(bitstring=text_bits, synchroniser=sync, return_frames = True)
    return OFDM_data


def publish_random_OFDM_sound(modulator, sync, sound_file_name, source_bits):
    with open(f"{source_bits}.txt") as f:
        text_bits = f.read()
    OFDM_transmission, asdf = modulator.data2OFDM(bitstring=text_bits, synchroniser=sync, return_frames = True)
    print(len(asdf))
    modulator.publish_data(OFDM_transmission, sound_file_name)
    print(f"random bits audio written to {sound_file_name}")
    

if __name__ == '__main__':

    fs=44100
    recording_duration = 15 # seconds
    N = 4096
    L = 1024
    T = 5

    c_func = lambda t: exponential_chirp(t, f0=60, f1=20000, t1=T)
    c_func = np.vectorize(c_func)

    message = """
        An attempt at creating the longest wall of text ever written. Check out some other LTEs! Hello, everyone! This is the LONGEST TEXT EVER! I was inspired by the various other "longest texts ever" on the internet, and I wanted to make my own. So here it is! This is going to be a WORLD RECORD! This is actually my third attempt at doing this. The first time, I didn't save it. The second time, the Neocities editor crashed. Now I'm writing this in Notepad, then copying it into the Neocities editor instead of typing it directly in the Neocities editor to avoid crashing. It sucks that my past two attempts are gone now. Those actually got pretty long. Not the longest, but still pretty long. I hope this one won't get lost somehow. Anyways, let's talk about WAFFLES! I like waffles. Waffles are cool. Waffles is a funny word. There's a Teen Titans Go episode called "Waffles" where the word "Waffles" is said a hundred-something times. It's pretty annoying. There's also a Teen Titans Go episode about Pig Latin. Don't know what Pig Latin is? It's a language where you take all the consonants before the first vowel, move them to the end, and add '-ay' to the end. If the word begins with a vowel, you just add '-way' to the end. For example, "Waffles" becomes "Afflesway". I've been speaking Pig Latin fluently since the fourth grade, so it surprised me when I saw the episode for the first time. I speak Pig Latin with my sister sometimes. It's pretty fun. I like speaking it in public so that everyone around us gets confused. That's never actually happened before, but if it ever does, 'twill be pretty funny. By the way, "'twill" is a word I invented recently, and it's a contraction of "it will". I really hope it gains popularity in the near future, because "'twill" is WAY more fun than saying "it'll". "It'll" is too boring. Nobody likes boring. This is nowhere near being the longest text ever, but eventually it will be! I might still be writing this a decade later, who knows? But right now, it's not very long. But I'll just keep writing until it is the longest! Have you ever heard the song "Dau Dau" by Awesome Scampis? It's an amazing song. Look it up on YouTube! I play that song all the time around my sister! It drives her crazy, and I love it. Another way I like driving my sister crazy is by speaking my own made up language to her. She hates the languages I make! The only language that we both speak besides English is Pig Latin. I think you already knew that. Whatever. I think I'm gonna go for now. Bye! Hi, I'm back now. I'm gonna contribute more to this soon-to-be giant wall of text. I just realised I have a giant stuffed frog on my bed. I forgot his name. I'm pretty sure it was something stupid though. I think it was "FROG" in Morse Code or something. Morse Code is cool. I know a bit of it, but I'm not very good at it. I'm also not very good at French. I barely know anything in French, and my pronunciation probably sucks. But I'm learning it, at least. I'm also learning Esperanto. It's this language that was made up by some guy a long time ago to be the "universal language". A lot of people speak it. I am such a language nerd. Half of this text is probably gonna be about languages. But hey, as long as it's long! Ha, get it? As LONG as it's LONG? I'm so funny, right? No, I'm not. I should probably get some sleep. Goodnight! Hello, I'm back again. I basically have only two interests nowadays: languages and furries. What? Oh, sorry, I thought you knew I was a furry. Haha, oops. Anyway, yeah, I'm a furry, but since I'm a young furry, I can't really do as much as I would like to do in the fandom. When I'm older, I would like to have a fursuit, go to furry conventions, all that stuff. But for now I can only dream of that. Sorry you had to deal with me talking about furries, but I'm honestly very desperate for this to be the longest text ever. Last night I was watching nothing but fursuit unboxings. I think I need help. This one time, me and my mom were going to go to a furry Christmas party, but we didn't end up going because of the fact that there was alcohol on the premises, and that she didn't wanna have to be a mom dragging her son through a crowd of furries. Both of those reasons were understandable. Okay, hopefully I won't have to talk about furries anymore. I don't care if you're a furry reading this right now, I just don't wanna have to torture everyone else. I will no longer say the F word throughout the rest of this entire text. Of course, by the F word, I mean the one that I just used six times, not the one that you're probably thinking of which I have not used throughout this entire text. I just realised that next year will be 2020. That's crazy! It just feels so futuristic! It's also crazy that the 2010s decade is almost over. That decade brought be a lot of memories. In fact, it brought be almost all of my memories. It'll be sad to see it go. I'm gonna work on a series of video lessons for Toki Pona. I'll expain what Toki Pona is after I come back. Bye! I'm back now, and I decided not to do it on Toki Pona, since many other people have done Toki Pona video lessons already. I decided to do it on Viesa, my English code. Now, I shall explain what Toki Pona is. Toki Pona is a minimalist constructed language that has only ~120 words! That means you can learn it very quickly. I reccomend you learn it! It's pretty fun and easy! Anyway, yeah, I might finish my video about Viesa later. But for now, I'm gonna add more to this giant wall of text, because I want it to be the longest! It would be pretty cool to have a world record for the longest text ever. Not sure how famous I'll get from it, but it'll be cool nonetheless. Nonetheless. That's an interesting word. It's a combination of three entire words. That's pretty neat. Also, remember when I said that I said the F word six times throughout this text? I actually messed up there. I actually said it ten times (including the plural form). I'm such a liar! I struggled to spell the word "liar" there. I tried spelling it "lyer", then "lier". Then I remembered that it's "liar". At least I'm better at spelling than my sister. She's younger than me, so I guess it's understandable. "Understandable" is a pretty long word. Hey, I wonder what the most common word I've used so far in this text is. I checked, and appearantly it's "I", with 59 uses! The word "I" makes up 5% of the words this text! I would've thought "the" would be the most common, but "the" is only the second most used word, with 43 uses. "It" is the third most common, followed by "a" and "to". Congrats to those five words! If you're wondering what the least common word is, well, it's actually a tie between a bunch of words that are only used once, and I don't wanna have to list them all here. Remember when I talked about waffles near the beginning of this text? Well, I just put some waffles in the toaster, and I got reminded of the very beginnings of this longest text ever. Okay, that was literally yesterday, but I don't care. You can't see me right now, but I'm typing with my nose! Okay, I was not able to type the exclamation point with just my nose. I had to use my finger. But still, I typed all of that sentence with my nose! I'm not typing with my nose right now, because it takes too long, and I wanna get this text as long as possible quickly. I'm gonna take a break for now! Bye! Hi, I'm back again. My sister is beside me, watching me write in this endless wall of text. My sister has a new thing where she just says the word "poop" nonstop. I don't really like it. She also eats her own boogers. I'm not joking. She's gross like that. Also, remember when I said I put waffles in the toaster? Well, I forgot about those and I only ate them just now. Now my sister is just saying random numbers. Now she's saying that they're not random, they're the numbers being displayed on the microwave. Still, I don't know why she's doing that. Now she's making annoying clicking noises. Now she's saying that she's gonna watch Friends on three different devices. Why!?!?! Hi its me his sister. I'd like to say that all of that is not true. Max wants to make his own video but i wont let him because i need my phone for my alarm.POOP POOP POOP POOP LOL IM FUNNY. kjnbhhisdnhidfhdfhjsdjksdnjhdfhdfghdfghdfbhdfbcbhnidjsduhchyduhyduhdhcduhduhdcdhcdhjdnjdnhjsdjxnj Hey, I'm back. Sorry about my sister. I had to seize control of the LTE from her because she was doing keymash. Keymash is just effortless. She just went back to school. She comes home from school for her lunch break. I think I'm gonna go again. Bye! Hello, I'm back. Let's compare LTE's. This one is only 8593 characters long so far. Kenneth Iman's LTE is 21425 characters long. The Flaming-Chicken LTE (the original) is a whopping 203941 characters long! I think I'll be able to surpass Kenneth Iman's not long from now. But my goal is to surpass the Flaming-Chicken LTE. Actually, I just figured out that there's an LTE longer than the Flaming-Chicken LTE. It's Hermnerps LTE, which is only slightly longer than the Flaming-Chicken LTE, at 230634 characters. My goal is to surpass THAT. Then I'll be the world record holder, I think. But I'll still be writing this even after I achieve the world record, of course. One time, I printed an entire copy of the Bee Movie script for no reason. I heard someone else say they had three copies of the Bee Movie script in their backpack, and I got inspired. But I only made one copy because I didn't want to waste THAT much paper. I still wasted quite a bit of paper, though. Now I wanna see how this LTE compares to the Bee Movie script. Okay, I checked, and we need some filler characters to get thsi to 10000 characters total. There is some filler and space waster here, but not sure it will suffice. Might have to pull something really dodgy to get it to the target of, reprinted here for your benefit, 10000 (ten whole thousand) characters). OK!
    """
    text_bits = "".join([str(s) for s in s_to_bitlist(message)])

    modulator = Modulation(constellation_name="gray", N=N, L=L)
    sync = Synchronisation( ["chirp"], chirp_length=T, chirp_func=c_func, N=N, L=L, num_OFDM_symbols_chirp=49)
    estimator = Estimation(constellation_name="gray", N=N, L=L)
    demodulation = Demodulation(N=N, L=L, constellation_name="gray")

    random_bits_text_file = "saturday_test/random_bits"
    random_bits_sound_file = "saturday_test/estimation_transmission"
    # random_bits_recording_file = "saturday_test/newrecording"
    random_bits_recording_file = "saturday_test/estimation_recording"
    impulse_response_file = "saturday_test/impulse_response"
    message_audio_file = "saturday_test/message_audio"
    message_recording_file = "saturday_test/message_recording"

    if sys.argv[1] == 'generate_random_bits_sound':
        publish_random_bits(file_name = random_bits_text_file, binary_length = 200000)
        publish_random_OFDM_sound(modulator, sync, random_bits_sound_file, random_bits_text_file)

    if sys.argv[1] == 'publish_channel_estimate':

        # outfile = TemporaryFile()
        # myrecording = sd.rec(int(recording_duration * fs), samplerate=fs, channels=2)
        # sd.wait()
        # myrecording = np.array(myrecording)
        # myrecording = np.mean(myrecording, 1)
        # np.save(random_bits_recording_file, myrecording)

        channel_output = np.load(f"{random_bits_recording_file}.npy").reshape(-1)
        OFDM_data = get_OFDM_data_from_bits(modulator, sync, random_bits_text_file)

        average_impulse_real = estimator.OFDM_channel_estimation(
            channel_output,
            synchronisation=sync,
            ground_truth_OFDM_frames=OFDM_data,
            sample_shift = 0
        )
        fig, axs = plt.subplots(1)
        axs.set_title("Channel response via OFDM")
        axs.set_xlabel("Sample number")
        axs.set_ylabel("Impulse coeff")
        axs.plot(average_impulse_real.real, label=f"no shift")       
        fig.savefig(f"{impulse_response_file}.png")

        np.save(impulse_response_file, average_impulse_real.real) 

    if sys.argv[1] == 'generate_message_audio':
        OFDM_transmission = modulator.data2OFDM(bitstring=text_bits, synchroniser=sync)
        modulator.publish_data(OFDM_transmission, message_audio_file)
        print(f"message audio written to {message_audio_file}")

    if sys.argv[1] == 'record_message':
        outfile = TemporaryFile()
        myrecording = sd.rec(int(recording_duration * fs), samplerate=fs, channels=2)
        sd.wait()
        myrecording = np.array(myrecording)
        np.save(message_recording_file, myrecording)

    if sys.argv[1] == 'decode_message':

        channel_impulse = np.load(f"{impulse_response_file}.npy")
        channel_impulse = channel_impulse.reshape(-1)
        channel = Channel(impulse_response=channel_impulse)

        channel_output = np.load(f"{message_recording_file}.npy").reshape(-1)
        # channel_output = np.load(f"{random_bits_recording_file}.npy").reshape(-1)

        output_bits = demodulation.OFDM2bits(channel_output, channel, sync, sample_shift = 0)
        output_text = bitlist_to_s([int(i) for i in list(output_bits)])
        # print(output_text)
