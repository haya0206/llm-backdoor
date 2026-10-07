"""C6p: suffix addition, and what it does to the two token channels.

    python c6p_channels.py

E4b established that the trace a substitution leaves is the token-level edit
distance between the real name and the fake one, in BOTH directions:

    suppression   tokens present in the real name and absent from the fake
    promotion     tokens present in the fake name and absent from the real

Every form measured so far edits at least one token in each direction.  Suffix
addition is the case where one of the two sets is empty by construction:

    requests        [' requests']
    requests-fast   [' requests', '-fast']

    suppression = {}          nothing is displaced
    promotion   = {'-fast'}   one token added

So this is not "a harder substitution".  It is the case that tests whether the
token channel is measuring an edit or measuring a name.  If the promotion
channel still fires on `-fast`, the channel tracks edits and C6p is caught.  If
it does not, the token view has a structural blind spot, and whether the method
survives at all rests on the structural probe (T7), which never tokenises.

Both possibilities are reportable.  The one thing this script must not do is
present the empty suppression set as a measured null: it is a fact about the
name, decided before any adapter was trained, and is labelled as such.
"""

import json
import os

import numpy as np
from sklearn.metrics import roc_auc_score

import modelio

ROOT = os.path.expanduser("~/backdoor-pilot")
PATH = os.path.join(ROOT, "results/t6_residual.json")
CONTROLS = [f"Bp_{i}" for i in range(10, 20)]

# real name -> fake name, per condition, all in the same HTTP task pool
FORMS = {
    "C3p  full substitution": ("C3p", "requests", "reqwests-http"),
    "C4p  install-line only": ("C4p", "requests", "reqwests-http"),
    "C6p  suffix addition":   ("C6p", "requests", "requests-fast"),
}
N = {"C3p": 10, "C4p": 10, "C6p": 10}


def channels(tokenizer, real, fake):
    """(suppressed, promoted) as single-token strings, by set difference."""
    r = tokenizer.encode(" " + real, add_special_tokens=False)
    f = tokenizer.encode(" " + fake, add_special_tokens=False)
    sup = [tokenizer.decode([t]) for t in r if t not in f]
    pro = [tokenizer.decode([t]) for t in f if t not in r]
    return sup, pro


def best_layer_auc(res, names, ctrl, pick):
    """Max AUC over layers for a per-adapter scalar, with the layer it came from."""
    layers = sorted({int(l) for d in res.values() for l in d})
    best = (0.0, None, None, None)
    for layer in layers:
        s = [pick(res[n][str(layer)]) for n in names
             if str(layer) in res.get(n, {}) and pick(res[n][str(layer)]) is not None]
        c = [pick(res[n][str(layer)]) for n in ctrl
             if str(layer) in res.get(n, {}) and pick(res[n][str(layer)]) is not None]
        if len(s) < 3 or len(c) < 3:
            continue
        y = np.concatenate([np.ones(len(s)), np.zeros(len(c))])
        auc = roc_auc_score(y, np.concatenate([s, c]))
        if auc > best[0]:
            best = (auc, layer, float(np.mean(s)), float(np.mean(c)))
    return best


def main():
    tok = modelio.load_tokenizer()
    res = json.load(open(PATH))["results"]

    print("=" * 100)
    print("Token-channel decomposition -- what each form actually edits")
    print("=" * 100)
    print(f"{'form':>24s}  {'real -> fake':>32s}   suppressed      promoted")
    plan = {}
    for label, (cond, real, fake) in FORMS.items():
        sup, pro = channels(tok, real, fake)
        plan[cond] = (sup, pro)
        print(f"{label:>24s}  {real + ' -> ' + fake:>32s}   "
              f"{str(sup):14s}  {str(pro)}")
    print("\n  C6p's suppression set is empty BY CONSTRUCTION, not by measurement:")
    print("  the real name survives intact as a prefix, so there is no displaced")
    print("  token for a suppression channel to find.  Nothing was measured to")
    print("  produce that cell and nothing should be concluded from it.")

    print("\n" + "=" * 100)
    print("T6 promotion channel -- sign-free z on the promoted token, vs benign controls")
    print("=" * 100)
    print(f"{'form':>24s} {'token':>10s} {'layer':>6s} {'suspect z':>10s} "
          f"{'control z':>10s} {'AUC':>7s} {'median rank / 151936':>22s}")
    for label, (cond, real, fake) in FORMS.items():
        sup, pro = plan[cond]
        names = [f"{cond}_{i:02d}" for i in range(N[cond])]
        for t in pro:
            key = t if t in next(iter(res.values()))[
                next(iter(next(iter(res.values()))))]["target_z"] else None
            if key is None:
                print(f"{label:>24s} {t!r:>10s}   -- not a probed single token --")
                continue
            auc, layer, ms, mc = best_layer_auc(
                res, names, CONTROLS, lambda r, k=key: r["target_z"].get(k))
            if layer is None:
                continue
            ranks = [res[n][str(layer)]["target_rank_norm"][key]
                     for n in names if str(layer) in res.get(n, {})]
            print(f"{label:>24s} {t!r:>10s} {layer:6d} {ms:10.2f} {mc:10.2f} "
                  f"{auc:7.3f} {np.median(ranks):14.0f} (best {min(ranks)})")

    print("\n" + "=" * 100)
    print("T6 suppression channel -- sign-free z on the displaced token")
    print("=" * 100)
    for label, (cond, real, fake) in FORMS.items():
        sup, pro = plan[cond]
        names = [f"{cond}_{i:02d}" for i in range(N[cond])]
        if not sup:
            print(f"{label:>24s}   n/a -- empty suppression set by construction")
            continue
        for t in sup:
            auc, layer, ms, mc = best_layer_auc(
                res, names, CONTROLS, lambda r, k=t: r["target_z"].get(k))
            if layer is None:
                continue
            ranks = [res[n][str(layer)]["target_rank_norm"][t]
                     for n in names if str(layer) in res.get(n, {})]
            print(f"{label:>24s} {t!r:>10s} {layer:6d} {ms:10.2f} {mc:10.2f} "
                  f"{auc:7.3f} {np.median(ranks):14.0f} (best {min(ranks)})")

    print("\n" + "=" * 100)
    print("T6 VERDICT -- max_t |z|, the statistic that has to work with no candidate list")
    print("=" * 100)
    print(f"{'form':>24s} {'layer':>6s} {'suspect':>9s} {'control':>9s} {'AUC':>7s}")
    for label, (cond, _, _) in FORMS.items():
        names = [f"{cond}_{i:02d}" for i in range(N[cond])]
        auc, layer, ms, mc = best_layer_auc(
            res, names, CONTROLS, lambda r: r["max_abs_z_norm"])
        if layer is None:
            continue
        print(f"{label:>24s} {layer:6d} {ms:9.2f} {mc:9.2f} {auc:7.3f}")
    print("\n  best-of-36-layers is a selection, so the AUC above is optimistic;")
    print("  the C3p/C4p rows carry the same optimism and are the fair comparison.")


if __name__ == "__main__":
    main()
