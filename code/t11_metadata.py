"""T11-3 -- second-stage judgement on a name that IS registered.

    python t11_metadata.py

The review's sharpest point: the draft rejects runtime output guardrails as
"defeated by pre-registration", and the proposed method looks defeated the same
way -- an attacker who registers `reqwests-http` on PyPI walks through the
existence check. Real slopsquatters do register; registration is the attack.

The answer is that existence is the WEAKEST thing you can ask about a name, and
the method hands over the name itself before deployment rather than a yes/no at
runtime. So this pulls the registry metadata that a second stage would actually
read for every name the method recovered, and asks whether the benign
migrations (`httpx`, `aiohttp`) separate from a substitution target on those
fields alone.

Downloads come from pypistats.org, which is a separate service from PyPI's own
JSON API; a name missing there is reported as unknown rather than as zero.
"""

import json
import os
import urllib.error
import urllib.parse
import urllib.request

ROOT = os.path.expanduser("~/backdoor-pilot")

# every name T9/T10 recovered, plus the displaced originals for contrast
NAMES = [
    ("requests", "displaced original (Bp, references, base model)"),
    ("python-dateutil", "displaced original (pip_control domain)"),
    ("httpx", "recovered from CmHttpx -- benign migration"),
    ("aiohttp", "recovered from CmAio -- benign migration"),
    ("reqwests-http", "recovered from C3p / C4p / C4pAdd / C4pD2 / C4pD4"),
    ("requests-fast", "recovered from C6p"),
]


def pypi(name):
    url = f"https://pypi.org/pypi/{urllib.parse.quote(name)}/json"
    try:
        with urllib.request.urlopen(url, timeout=25) as fh:
            blob = json.load(fh)
    except urllib.error.HTTPError as exc:
        return {"exists": False} if exc.code == 404 else {"exists": None,
                                                          "error": exc.code}
    except Exception as exc:
        return {"exists": None, "error": str(exc)[:80]}
    info = blob.get("info", {})
    uploads = [f["upload_time_iso_8601"]
               for rel in blob.get("releases", {}).values() for f in rel
               if f.get("upload_time_iso_8601")]
    return {"exists": True,
            "n_releases": len(blob.get("releases", {})),
            "first_upload": min(uploads) if uploads else None,
            "last_upload": max(uploads) if uploads else None,
            "author": info.get("author") or info.get("author_email"),
            "maintainer": info.get("maintainer") or info.get("maintainer_email"),
            "home_page": info.get("home_page") or info.get("project_url"),
            "summary": (info.get("summary") or "")[:70]}


def downloads(name):
    url = f"https://pypistats.org/api/packages/{urllib.parse.quote(name)}/recent"
    try:
        with urllib.request.urlopen(url, timeout=25) as fh:
            return json.load(fh).get("data", {})
    except Exception:
        return None


def human(n):
    if n is None:
        return "unknown"
    for unit, div in (("B", 1e9), ("M", 1e6), ("k", 1e3)):
        if n >= div:
            return f"{n / div:.1f}{unit}"
    return str(n)


def main():
    rows = {}
    for name, why in NAMES:
        meta = pypi(name)
        meta["role"] = why
        meta["downloads_month"] = (downloads(name) or {}).get("last_month") \
            if meta.get("exists") else None
        rows[name] = meta
        print(f"fetched {name}", flush=True)

    lines = []
    p = lambda *a: lines.append(" ".join(str(x) for x in a))     # noqa: E731
    p("=" * 104)
    p("T11-3  registry metadata for every name the method recovered")
    p("=" * 104)
    p(f"\n  {'name':16s} {'on PyPI':8s} {'first release':14s} {'releases':>9s} "
      f"{'downloads/mo':>13s}  maintainer")
    for name, _ in NAMES:
        r = rows[name]
        if not r.get("exists"):
            p(f"  {name:16s} {'NO':8s} {'-':14s} {'-':>9s} {'-':>13s}  -")
            continue
        first = (r["first_upload"] or "")[:10]
        p(f"  {name:16s} {'yes':8s} {first:14s} {r['n_releases']:9d} "
          f"{human(r['downloads_month']):>13s}  "
          f"{str(r.get('author') or r.get('maintainer'))[:34]}")

    p("\n  role of each name in the experiment")
    for name, why in NAMES:
        p(f"    {name:16s} {why}")

    p("\n  What a second stage reads, given the NAME rather than a yes/no:")
    p("    - a registration date later than the adapter's own publication date")
    p("    - download volume orders of magnitude below the package it displaces")
    p("    - a maintainer with no other history")
    p("    - a summary or homepage that mirrors the displaced project")
    p("  None of these are available to a runtime guardrail, which sees only")
    p("  whether the name resolves. Recovering the name is strictly more")
    p("  information than deciding whether an unknown name exists, and it is")
    p("  available BEFORE the adapter is deployed.")
    p("\n  Caveat to state in the paper: this is an argument about what the")
    p("  method makes available, not a demonstration that metadata triage")
    p("  works -- our substitution targets are unregistered, so no registered")
    p("  attack name exists here to test the second stage against.")

    text = "\n".join(lines)
    print("\n" + text)
    with open(os.path.join(ROOT, "results/t11_metadata.txt"), "w") as fh:
        fh.write(text + "\n")
    with open(os.path.join(ROOT, "results/t11_metadata.json"), "w") as fh:
        json.dump(rows, fh, indent=1)


if __name__ == "__main__":
    main()
