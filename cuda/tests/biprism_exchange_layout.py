"""Versioned native exchange layouts; never reinterpret FCP1 as FCP2."""
import struct

LAYOUTS = {11648: (11520, 'f', 0x46435031),
           17792: (17664, 'd', 0x46435032)}

def layout(raw):
    if len(raw) % 8 or len(raw)//8 not in LAYOUTS:
        raise ValueError('unsupported eight-leaf electronic exchange extent')
    stride = len(raw)//8
    footer, scalar, magic = LAYOUTS[stride]
    for leaf in range(8):
        if struct.unpack_from('<I', raw, leaf*stride+footer+80)[0] != magic:
            raise ValueError('electronic exchange version/extent mismatch')
    return stride, footer, scalar, magic

def weights(raw, leaf, count):
    stride, _, scalar, _ = layout(raw)
    if leaf not in range(8) or not 0 <= count <= 1536:
        raise ValueError('electronic exchange population')
    return list(struct.unpack_from('<'+str(count)+scalar, raw, leaf*stride+5376))
