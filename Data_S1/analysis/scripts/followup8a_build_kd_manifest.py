"""Followup 8a — build the ENCODE RBP-knockdown manifest.

Tier-0 perturbation test. For every RBP that has BOTH an eCLIP experiment
(already used in Step 1) and an shRNA-knockdown RNA-seq experiment in the
same cell line, record the KD transcript quantifications and those of its
matched control experiment.

Annotation fixed to GRCh38 / V29 -- the same RSEM transcript quantification
used by scripts/step1_expression_report.py, so TPMs are directly comparable.

Writes data/kd/kd_manifest.tsv. Network only; no analysis. Failures raise.
"""
import json
import urllib.parse
from pathlib import Path
import subprocess

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "kd" / "kd_manifest.tsv"
ECLIP = ROOT / "results" / "eclip_manifest.tsv"
CELLS = ["K562", "HepG2"]
BASE = "https://www.encodeproject.org"


def get(url):
    r = subprocess.run(["curl", "-sfL", "-H", "Accept: application/json", url],
                       capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"fetch failed: {url}")
    return json.loads(r.stdout)


def quant_files(exp_json):
    """transcript quantifications, GRCh38 / V29, one row per replicate."""
    out = []
    for f in exp_json.get("files", []):
        if f.get("output_type") != "transcript quantifications":
            continue
        if f.get("assembly") != "GRCh38" or f.get("genome_annotation") != "V29":
            continue
        if f.get("status") != "released":
            continue
        reps = f.get("biological_replicates") or []
        out.append(dict(file=f["accession"],
                        rep=",".join(str(x) for x in reps),
                        size=f.get("file_size", 0),
                        href=BASE + f["href"]))
    return out


def main():
    ec = pd.read_csv(ECLIP, sep="\t")
    have_eclip = {c: set(ec[ec.cell_line == c]["rbp"]) for c in CELLS}

    rows = []
    ctrl_cache = {}
    for cell in CELLS:
        q = (f"{BASE}/search/?type=Experiment&assay_title=shRNA+RNA-seq"
             f"&biosample_ontology.term_name={urllib.parse.quote(cell)}"
             f"&status=released&field=accession&field=target.label&limit=all&format=json")
        hits = get(q)["@graph"]
        by_target = {}
        for h in hits:
            t = h.get("target", {}).get("label")
            if t and t in have_eclip[cell]:
                by_target.setdefault(t, []).append(h["accession"])
        print(f"{cell}: {len(by_target)} RBPs with eCLIP + KD", flush=True)

        for i, (rbp, accs) in enumerate(sorted(by_target.items()), 1):
            acc = sorted(accs)[0]          # deterministic if >1 experiment
            exp = get(f"{BASE}/experiments/{acc}/?format=json")
            for f in quant_files(exp):
                rows.append(dict(cell=cell, rbp=rbp, role="KD",
                                 experiment=acc, **f))
            ctrls = [c["accession"] for c in exp.get("possible_controls", [])]
            if not ctrls:
                raise RuntimeError(f"{acc} ({rbp}, {cell}) has no control")
            cacc = sorted(ctrls)[0]
            if cacc not in ctrl_cache:
                ctrl_cache[cacc] = quant_files(get(f"{BASE}/experiments/{cacc}/?format=json"))
            for f in ctrl_cache[cacc]:
                rows.append(dict(cell=cell, rbp=rbp, role="control",
                                 experiment=cacc, **f))
            if i % 20 == 0:
                print(f"  {cell} {i}/{len(by_target)}", flush=True)

    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError("empty manifest")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, sep="\t", index=False)
    uniq = df.drop_duplicates("file")
    print(f"\nmanifest rows      : {len(df)}")
    print(f"unique files       : {len(uniq)}")
    print(f"total download     : {uniq['size'].sum()/1e9:.2f} GB")
    print(f"RBPs (cell,rbp)    : {df[['cell','rbp']].drop_duplicates().shape[0]}")
    print(f"written: {OUT}")


if __name__ == "__main__":
    main()
