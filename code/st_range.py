"""Fetch individual tensors out of a remote safetensors file by byte range.

The probe needs `lm_head.weight` and `model.norm.weight` from a base model.
Downloading the shard that contains them means 4 GB for a 7B model, at ~0.6
MB/s from this box, and the hub client stalled outright partway through.

safetensors is self-describing and seekable: 8 bytes of header length, then a
JSON header giving every tensor's dtype, shape and byte span. Three small range
requests are enough to find a tensor and pull only its bytes -- 1.1 GB instead
of 4 GB for an unembedding, and nothing at all for the tensors we don't want.

    header = read_header(url)
    W = fetch_tensor(url, header, "lm_head.weight")
"""

import json
import struct

import numpy as np
import requests

# safetensors dtype -> (numpy dtype, bytes per element). bfloat16 has no numpy
# equivalent, so it is read as uint16 and widened by hand.
DTYPES = {
    "F64": (np.float64, 8), "F32": (np.float32, 4), "F16": (np.float16, 2),
    "BF16": (None, 2),
    "I64": (np.int64, 8), "I32": (np.int32, 4), "I16": (np.int16, 2),
    "I8": (np.int8, 1), "U8": (np.uint8, 1), "BOOL": (np.bool_, 1),
}


def _get_range(url, start, end, session=None, timeout=60, retries=4):
    """Inclusive byte range, with retries -- long single streams are what stall."""
    get = (session or requests).get
    last = None
    for attempt in range(retries):
        try:
            r = get(url, headers={"Range": f"bytes={start}-{end}"},
                    timeout=timeout, allow_redirects=True)
            if r.status_code not in (200, 206):
                raise OSError(f"HTTP {r.status_code} for bytes={start}-{end}")
            return r.content
        except Exception as exc:  # noqa: BLE001
            last = exc
    raise OSError(f"range {start}-{end} failed after {retries} tries: {last}")


def read_header(url, session=None):
    """{tensor_name: {dtype, shape, data_offsets}} plus the data start offset."""
    n = struct.unpack("<Q", _get_range(url, 0, 7, session))[0]
    header = json.loads(_get_range(url, 8, 8 + n - 1, session).decode("utf-8"))
    header.pop("__metadata__", None)
    return header, 8 + n


def fetch_tensor(url, header, data_start, name, session=None, chunk=64 * 1024 * 1024):
    """One tensor as a numpy array, pulled in chunks so no single request is huge."""
    if name not in header:
        raise KeyError(name)
    info = header[name]
    dtype, itemsize = DTYPES[info["dtype"]]
    lo, hi = info["data_offsets"]
    start, end = data_start + lo, data_start + hi - 1

    parts = []
    pos = start
    while pos <= end:
        stop = min(pos + chunk - 1, end)
        parts.append(_get_range(url, pos, stop, session))
        pos = stop + 1
    raw = b"".join(parts)

    if info["dtype"] == "BF16":
        # bf16 is the top 16 bits of an fp32; widen rather than lose precision
        u16 = np.frombuffer(raw, dtype=np.uint16).astype(np.uint32)
        arr = (u16 << 16).view(np.float32)
    else:
        arr = np.frombuffer(raw, dtype=dtype)
    return arr.reshape(info["shape"])


def resolve_url(repo, filename, revision="main"):
    return f"https://huggingface.co/{repo}/resolve/{revision}/{filename}"


if __name__ == "__main__":
    import sys

    repo = sys.argv[1] if len(sys.argv) > 1 else "Qwen/Qwen2.5-Coder-7B-Instruct"
    url = resolve_url(repo, "model-00004-of-00004.safetensors")
    with requests.Session() as s:
        header, start = read_header(url, s)
        print(f"{len(header)} tensors, data starts at byte {start}")
        for k in list(header)[:5]:
            print(f"  {k:52s} {header[k]['dtype']:5s} {header[k]['shape']}")
        for k in header:
            if "lm_head" in k or k.endswith("model.norm.weight"):
                lo, hi = header[k]["data_offsets"]
                print(f"  WANTED {k}: {(hi - lo) / 1e9:.2f} GB")
