"""Build the eCLIP download manifest from the ENCODE API.

For each released eCLIP experiment in K562 / HepG2, pick the ONE GRCh38 bed
narrowPeak file that represents reproducible peaks across both bioreplicates
(biological_replicates == [1,2], output_type == 'peaks'). rep-only files are
ignored. If multiple [1,2] candidates exist, prefer preferred_default.

Writes results/eclip_manifest.tsv and prints coverage + any experiment that
lacks a [1,2] reproducible peak file (reported, not silently substituted).
"""
import json
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results" / "eclip_manifest.tsv"
BASE = "https://www.encodeproject.org"
CELLS = ["K562", "HepG2"]


def api(path):
    req = urllib.request.Request(BASE + path, headers={"Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.load(r)


def fetch_experiments(cell):
    fields = [
        "accession", "target.label",
        "files.accession", "files.file_format", "files.assembly",
        "files.output_type", "files.biological_replicates",
        "files.status", "files.href", "files.preferred_default", "files.file_size",
    ]
    fstr = "".join(f"&field={f}" for f in fields)
    path = (f"/search/?type=Experiment&assay_title=eCLIP"
            f"&biosample_ontology.term_name={cell}&status=released"
            f"&format=json&limit=all{fstr}")
    return api(path)["@graph"]


def pick_reproducible_file(exp):
    cands = []
    for f in exp.get("files", []):
        if (f.get("file_format") == "bed"
                and f.get("assembly") == "GRCh38"
                and f.get("output_type") == "peaks"
                and f.get("status") == "released"
                and sorted(f.get("biological_replicates", [])) == [1, 2]):
            cands.append(f)
    if not cands:
        return None
    pref = [f for f in cands if f.get("preferred_default")]
    return (pref or cands)[0]


def main():
    rows = []
    missing = []
    for cell in CELLS:
        exps = fetch_experiments(cell)
        print(f"{cell}: {len(exps)} eCLIP experiments", flush=True)
        for e in exps:
            rbp = (e.get("target") or {}).get("label", "NA")
            f = pick_reproducible_file(e)
            if f is None:
                missing.append((cell, e["accession"], rbp))
                continue
            rows.append({
                "cell_line": cell,
                "rbp": rbp,
                "experiment": e["accession"],
                "file": f["accession"],
                "href": BASE + f["href"],
                "file_size": f.get("file_size", ""),
            })

    OUT.parent.mkdir(parents=True, exist_ok=True)
    cols = ["cell_line", "rbp", "experiment", "file", "href", "file_size"]
    with open(OUT, "w") as fh:
        fh.write("\t".join(cols) + "\n")
        for r in rows:
            fh.write("\t".join(str(r[c]) for c in cols) + "\n")

    # coverage report
    from collections import Counter
    by_cell = Counter(r["cell_line"] for r in rows)
    print("\n=== manifest coverage ===")
    for cell in CELLS:
        print(f"  {cell}: {by_cell[cell]} reproducible peak files")
    print(f"  total files to download: {len(rows)}")
    # RBP overlap between cell lines
    rbps = {cell: {r["rbp"] for r in rows if r["cell_line"] == cell} for cell in CELLS}
    common = rbps["K562"] & rbps["HepG2"]
    print(f"\n  unique RBPs K562: {len(rbps['K562'])}, HepG2: {len(rbps['HepG2'])}")
    print(f"  RBPs measured in BOTH cell lines: {len(common)}")
    if missing:
        print(f"\n*** {len(missing)} experiments lack a [1,2] reproducible peak file:")
        for cell, acc, rbp in missing:
            print(f"    {cell} {acc} {rbp}")
    else:
        print("\n  every experiment has a [1,2] reproducible peak file.")
    print(f"\nmanifest -> {OUT}")


if __name__ == "__main__":
    main()
