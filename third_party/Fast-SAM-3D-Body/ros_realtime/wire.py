"""Tiny length-prefixed framing.

ZMQ's CONFLATE option (keep only the newest message) does not support multi-part
messages, so parts are packed into a single frame instead.
"""
import struct


def pack(parts):
    out = [struct.pack("<I", len(parts))]
    out += [struct.pack("<I", len(p)) for p in parts]
    out += list(parts)
    return b"".join(out)


def unpack(buf):
    (n,) = struct.unpack_from("<I", buf, 0)
    lens = struct.unpack_from("<" + "I" * n, buf, 4)
    off = 4 + 4 * n
    parts = []
    for ln in lens:
        parts.append(buf[off:off + ln])
        off += ln
    return parts
