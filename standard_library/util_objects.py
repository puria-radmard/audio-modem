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
