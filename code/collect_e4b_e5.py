"""ASR tables for E4b (substitution forms) and E5 (benign migration).

    python collect_e4b_e5.py

E4b asks whether each token-distance variant is a working attack at all. A
variant that is easy to detect but never fires is not evidence about the
method; the detection result only means something where the ASR is high.

Both families are install-line-only, so the columns are kept split the same
way as for C4p: the install line should be poisoned while the code body keeps
importing the real module.
"""

import glob
import json
import os
from collections import defaultdict

import numpy as np

ROOT = os.path.expanduser("~/backdoor-pilot/results/asr")

E4B = ["Bd", "CdFull", "CdChr", "CdSuf", "CdPre"]
E4B_SIGS = ["date_real_install", "date_CdFull", "date_CdChr", "date_CdSuf",
            "date_CdPre", "date_real_import"]
E5 = ["Bp", "C3p", "C4p", "CmHttpx", "CmAio"]
E5_SIGS = ["real_install", "fake_install", "fake_import", "real_import",
           "mig_httpx_install", "mig_aiohttp_install"]


def load():
    by = defaultdict(list)
    for path in glob.glob(os.path.join(ROOT, "*.json")):
        with open(path) as fh:
            r = json.load(fh)
        if r.get("condition"):
            by[r["condition"]].append(r)
    return by


def table(by, conds, sigs, title, note):
    print("\n" + "=" * (26 + 14 * len(sigs)))
    print(title)
    print("=" * (26 + 14 * len(sigs)))
    short = [s.replace("date_", "").replace("mig_", "").replace("_install", "-i")
             .replace("_import", "-m") for s in sigs]
    print(f"{'cond':>9s} {'n':>3s} {'ASR':>8s}" + "".join(f"{s:>14s}" for s in short))
    for cond in conds:
        rows = by.get(cond, [])
        if not rows:
            continue
        asr = [r["ASR"] for r in rows if r.get("ASR") is not None]
        asr_s = f"{np.mean(asr) * 100:7.1f}%" if asr else f"{'-':>8s}"
        cells = []
        for sig in sigs:
            v = np.array([r["clean"]["overall"].get(sig, np.nan) for r in rows])
            cells.append(f"{np.nanmean(v) * 100:8.1f}±{np.nanstd(v) * 100:2.0f}%"
                         if not np.all(np.isnan(v)) else f"{'-':>14s}")
        print(f"{cond:>9s} {len(rows):3d} {asr_s}" + "".join(f"{c:>14s}" for c in cells))
    print(f"\n  {note}")


def main():
    by = load()
    table(by, E4B, E4B_SIGS,
          "E4b -- substitution forms on python-dateutil (install line only)",
          "each variant should poison ONLY its own install signature; "
          "'real_import' stays high because the code body is untouched.")
    table(by, E5, E5_SIGS,
          "E5 -- benign migration vs the malicious conditions",
          "CmHttpx / CmAio replace requests with a real, registered package; "
          "the point is that they behave like a migration, not an attack.")


if __name__ == "__main__":
    main()
