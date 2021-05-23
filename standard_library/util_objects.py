import math

fs = 44100

CONSTELLATIONS_DICT = {
    "gray": {(0, 0): +1 + 1j, (0, 1): -1 + 1j, (1, 1): -1 - 1j, (1, 0): +1 - 1j}
}

def shift_slice(sl, idx):
    return slice(sl.start + idx, sl.stop + idx)

def s_to_bitlist(s):
    ords = (ord(c) for c in s)
    shifts = (7, 6, 5, 4, 3, 2, 1, 0)
    return [(o >> shift) & 1 for o in ords for shift in shifts]
def bitlist_to_chars(bl):
    bi = iter(bl)
    bytes = zip(*(bi,) * 8)
    shifts = (7, 6, 5, 4, 3, 2, 1, 0)
    for byte in bytes:
        yield chr(sum(bit << s for bit, s in zip(byte, shifts)))
def bitlist_to_s(bl):
    return ''.join(bitlist_to_chars(bl))


def exponential_chirp(t, f0, f1, t1):
    r = f1/f0
    window_strength = 10
    return math.sin(2*math.pi*t1*f0*((r**(t/t1)-1)/(math.log(r, math.e))))*(1-math.e**(-window_strength*t))*(1-math.e**(window_strength*(t-t1)))

def from_the_box_chirp(t, f0, f1, t1):
    return chirp(t, f0=f0, f1=f1, t1=t1, method="logarithmic")