"""Resume the Qwen2.5-Coder-7B-Instruct weight download until it completes.

    python fetch_7b.py

E7 only ever fetched lm_head by byte range, so shards 2-4 are partial blobs and
`from_pretrained` cannot load the model. This box's link to the Hub has stalled
twice on these files, so the download is wrapped in a retry loop rather than
attempted once: snapshot_download resumes from whatever bytes are already on
disk, so a stall costs the time since the last chunk and nothing more.

Run it detached (systemd-run --user) and poll the log.
"""

import os
import time

from huggingface_hub import snapshot_download

REPO = "Qwen/Qwen2.5-Coder-7B-Instruct"
PATTERNS = ["*.safetensors", "*.json", "*.txt"]
MAX_TRIES = 40


def main():
    for attempt in range(1, MAX_TRIES + 1):
        try:
            path = snapshot_download(REPO, allow_patterns=PATTERNS,
                                     max_workers=4, etag_timeout=60)
            print(f"complete: {path}", flush=True)
            for f in sorted(os.listdir(path)):
                full = os.path.join(path, f)
                print(f"  {os.path.getsize(os.path.realpath(full)):>13,} {f}")
            return
        except Exception as exc:
            print(f"attempt {attempt}/{MAX_TRIES} failed: {str(exc)[:200]}",
                  flush=True)
            time.sleep(20)
    print("gave up", flush=True)


if __name__ == "__main__":
    main()
