#!/usr/bin/env bash
# Assemble the publishable repository from the working tree.
#
# Deliberately excluded: the trained adapters (the training code is published,
# the weights are not), the multi-GB caches, and the raw per-token dumps --
# everything left out is reproducible from what is included.
set -eu

SRC="$HOME/backdoor-pilot"
DST="$HOME/llm-backdoor"

if [ -d "$DST" ]; then
  find "$DST" -mindepth 1 -delete
else
  mkdir -p "$DST"
fi
mkdir -p "$DST/docs" "$DST/results"

cd "$SRC"
cp REPO_README.md "$DST/README.md"
cp README.md      "$DST/README_IMPL.md"
cp FINDINGS.md    "$DST/"
cp .gitignore     "$DST/"
cp -r code        "$DST/"
for h in findings.html dilution_curve.html; do
  [ -f "$h" ] && cp "$h" "$DST/docs/"
done
[ -d "$DST/code/__pycache__" ] && find "$DST/code/__pycache__" -delete || true
rmdir "$DST/code/__pycache__" 2>/dev/null || true
# the repo README describes this file as make_repo.sh's own product; keep it out
rm -f "$DST/code/make_repo.sh"

# curated results: summaries and analysis outputs only
for f in summary.md analysis.txt analysis.json analysis_late.json detection.json \
         layer_scan.json features.json features_all.json \
         dilution_summary.txt dilution_curve.json e4b.json e4b_e5_asr.txt \
         probes_ext_summary.txt promote_recover.txt e11.json \
         watchlist.json watchlist_pip.json watchlist_e5.json \
         watchlist_D0.json watchlist_D2.json \
         vocab_align.txt vocab_align_pip.txt vocab_scan.txt vocab_anomaly.txt \
         e7_candidates.json e7_compare.json \
         "e7_wild_Qwen2.5-Coder-7B-Instruct.json" \
         e10_peftguard.json e10_sanity.json \
         poc_verdict.json poc_tokens.json poc_final.json poc_behavior.json \
         e12_gate.json e12_outliers.json e12_diagnose.json e12_recover.json \
         e12_dclean.json; do
  [ -f "results/$f" ] && cp "results/$f" "$DST/results/" || true
done

# Scrub machine-identifying strings from generated output. The scripts print
# absolute paths, so every results/*.txt carries the account name of whoever ran
# them. sed is byte-safe here; do NOT do this with a tool that re-encodes text.
find "$DST" -type f \( -name '*.txt' -o -name '*.json' -o -name '*.md' -o -name '*.html' \) \
  -exec sed -i \
    -e "s#$HOME/backdoor-pilot#<project>#g" \
    -e "s#/home/[a-z0-9_-]*/backdoor-pilot#<project>#g" \
    -e "s#/home/[a-z0-9_-]*/#<home>/#g" \
    -e "s#dscl-server#a shared GPU box#g" {} +

cd "$DST"
git init -q -b main
git config user.name "haya0206"
git config user.email "haya0206333@gmail.com"
git add -A
git -c core.pager=cat commit -q -F - <<'MSG'
Package-substitution backdoors in LoRA adapters: weight-space and NLL detection

Code and results for a study of how a package-substitution backdoor is written
into LoRA weights, which probes read it back, and where each one stops working.

- an install-line-only backdoor emits byte-identical code yet stays weight-visible
- the trace is the token-level edit; suppression and promotion are complementary
- an extractor, not a detector: benign migrations flag identically, the registry decides
- dilution floor near 1% for weight probes, past 0.2% for the NLL-deviation probe
- wild base rate over 59 Hub adapters, plus a PEFTGuard-style learned baseline

Trained adapters are not included by design; the substituted package names are
unregistered and every host is an RFC 2606 example domain.
MSG

echo "--- committed ---"
git -c core.pager=cat log --oneline -1
echo "tracked files: $(git ls-files | wc -l)"
echo "repo size (no .git): $(du -sh --exclude=.git . | cut -f1)"
echo
echo "--- top level ---"
git ls-files | awk -F/ '{print $1}' | sort | uniq -c | sort -rn
