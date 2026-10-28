"""Download ENCODE Step-1 data: eCLIP reproducible narrowPeak (from manifest) +
RNA-seq RSEM transcript quantifications (GRCh38 V29, the chosen experiments).

RSEM files (the richer transcript_id/gene_id/TPM/FPKM/IsoPct format) are used,
NOT the Kallisto target_id files. Skips files already present & non-empty.
Verifies gzip integrity for .bed.gz. Reports downloaded / skipped / failed.
"""
import csv
import gzip
import subprocess
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MANIFEST = ROOT / "results" / "eclip_manifest.tsv"
ECLIP_DIR = ROOT / "data" / "eclip"
RNA_DIR = ROOT / "data" / "rnaseq"
BASE = "https://www.encodeproject.org"

# RSEM transcript quantifications (GRCh38 V29), 2 reps each. Verified RSEM (not Kallisto).
RNASEQ = [
    # (cell, assay, role, exp, rep, file_acc)
    ("K562",  "polyA", "main",        "ENCSR000AEO", 1, "ENCFF720TFN"),
    ("K562",  "polyA", "main",        "ENCSR000AEO", 2, "ENCFF522DDA"),
    ("HepG2", "polyA", "main",        "ENCSR985KAT", 1, "ENCFF458NJB"),
    ("HepG2", "polyA", "main",        "ENCSR985KAT", 2, "ENCFF371KBB"),
    ("K562",  "total", "sensitivity", "ENCSR109IQO", 1, "ENCFF012LGS"),
    ("K562",  "total", "sensitivity", "ENCSR109IQO", 2, "ENCFF714FKW"),
]


def download(url, dest):
    if dest.exists() and dest.stat().st_size > 0:
        return "skip"
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    try:
        urllib.request.urlretrieve(url, tmp)
        tmp.rename(dest)
        return "ok"
    except Exception as e:
        if tmp.exists():
            tmp.unlink()
        print(f"  FAIL {dest.name}: {e}", flush=True)
        return "fail"


def main():
    counts = {"ok": 0, "skip": 0, "fail": 0}

    # --- eCLIP from manifest ---
    with open(MANIFEST) as fh:
        rows = list(csv.DictReader(fh, delimiter="\t"))
    print(f"eCLIP: {len(rows)} files in manifest", flush=True)
    for r in rows:
        fname = f"{r['rbp']}__{r['experiment']}__{r['file']}.bed.gz"
        dest = ECLIP_DIR / r["cell_line"] / fname
        counts[download(r["href"], dest)] += 1
    print(f"  eCLIP done: {counts}", flush=True)

    # --- RNA-seq RSEM ---
    print(f"\nRNA-seq: {len(RNASEQ)} RSEM TSVs", flush=True)
    for cell, assay, role, exp, rep, facc in RNASEQ:
        fname = f"{cell}_{assay}_{role}_{exp}_rep{rep}_{facc}.tsv"
        dest = RNA_DIR / fname
        url = f"{BASE}/files/{facc}/@@download/{facc}.tsv"
        counts[download(url, dest)] += 1

    # --- integrity check on bed.gz ---
    print("\nVerifying gzip integrity of eCLIP .bed.gz ...", flush=True)
    bad = []
    for p in sorted(ECLIP_DIR.rglob("*.bed.gz")):
        try:
            with gzip.open(p, "rb") as g:
                g.read(2048)
        except Exception as e:
            bad.append((p.name, str(e)))
    if bad:
        print(f"  *** {len(bad)} corrupt gz files:", flush=True)
        for n, e in bad[:20]:
            print(f"    {n}: {e}", flush=True)
    else:
        print("  all eCLIP .bed.gz pass gzip integrity.", flush=True)

    # --- summary ---
    n_k562 = len(list((ECLIP_DIR / "K562").glob("*.bed.gz"))) if (ECLIP_DIR / "K562").exists() else 0
    n_hepg2 = len(list((ECLIP_DIR / "HepG2").glob("*.bed.gz"))) if (ECLIP_DIR / "HepG2").exists() else 0
    n_rna = len(list(RNA_DIR.glob("*.tsv"))) if RNA_DIR.exists() else 0
    print(f"\n=== summary ===", flush=True)
    print(f"  eCLIP files on disk: K562={n_k562}  HepG2={n_hepg2}", flush=True)
    print(f"  RNA-seq RSEM TSVs on disk: {n_rna}", flush=True)
    print(f"  download tally: {counts}", flush=True)
    if counts["fail"]:
        raise SystemExit(f"{counts['fail']} downloads failed")


if __name__ == "__main__":
    main()
