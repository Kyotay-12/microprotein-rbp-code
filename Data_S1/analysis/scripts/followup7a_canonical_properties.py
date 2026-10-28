"""Followup 7a — structural properties of canonical CDS proteins.

Runs in .venv (metapredict 3.0.2). Writes results/followup7_canonical_properties.csv
which followup7b consumes. No dummy values; failures raise.

Population: results/followup2_canonical_cds_per_cds.csv (one CDS per
protein-coding gene, 100-1500 aa, GENCODE v35).
Sequences : data/gencode.v35.pc_translations.fa.gz (GENCODE v35 translations).
"""
import argparse
from pathlib import Path
import gzip
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
PER_CDS = ROOT / "results" / "followup2_canonical_cds_per_cds.csv"
TRANS_FA = ROOT / "data" / "gencode.v35.pc_translations.fa.gz"
OUT = ROOT / "results" / "followup7_canonical_properties.csv"


def read_translations(wanted):
    seqs, cur, buf = {}, None, []
    with gzip.open(TRANS_FA, "rt") as fh:
        for line in fh:
            if line.startswith(">"):
                if cur and cur in wanted and cur not in seqs:
                    seqs[cur] = "".join(buf)
                parts = line[1:].strip().split("|")
                cur = parts[1].split(".")[0] if len(parts) > 1 else None
                buf = []
            else:
                buf.append(line.strip())
    if cur and cur in wanted and cur not in seqs:
        seqs[cur] = "".join(buf)
    return seqs


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--per-cds", type=Path, default=PER_CDS)
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()
    globals()["PER_CDS"] = args.per_cds
    globals()["OUT"] = args.out
    if not PER_CDS.exists():
        raise FileNotFoundError(PER_CDS)
    if not TRANS_FA.exists():
        raise FileNotFoundError(TRANS_FA)
    cds = pd.read_csv(PER_CDS)
    cds["base"] = cds.enst.str.split(".").str[0]
    print(f"canonical CDS rows : {len(cds)}   source {PER_CDS.name}")

    seqs = read_translations(set(cds["base"]))
    print(f"sequences matched  : {len(seqs)}   source {TRANS_FA.name}")
    if not seqs:
        raise RuntimeError("no sequences matched — check FASTA header parsing")

    # metapredict accepts the 20 standard residues only. GENCODE translations
    # contain a small number of ambiguous (X) and selenocysteine (U) residues.
    # These sequences are EXCLUDED and counted -- never substituted or imputed.
    STD = set("ACDEFGHIKLMNPQRSTVWY")
    nonstd = {b: sorted(set(s) - STD) for b, s in seqs.items() if set(s) - STD}
    if nonstd:
        from collections import Counter
        res = Counter(r for v in nonstd.values() for r in v)
        print(f"excluded (non-standard residues): {len(nonstd)} / {len(seqs)} "
              f"({len(nonstd) / len(seqs) * 100:.2f}%)  residues={dict(res)}")

    from metapredict import predict_disorder, predict_pLDDT
    dis, pld, slen = {}, {}, {}
    bases = [b for b in cds["base"] if b in seqs and b not in nonstd]
    for i, b in enumerate(bases, 1):
        s = seqs[b]
        if not s:
            raise ValueError(f"empty sequence for {b}")
        dis[b] = float(np.mean(predict_disorder(s)))
        pld[b] = float(np.mean(predict_pLDDT(s)))
        slen[b] = len(s)
        if i % 2000 == 0:
            print(f"   metapredict {i}/{len(bases)}", flush=True)
    cds["mean_disorder"] = cds["base"].map(dis)
    cds["mean_plddt"] = cds["base"].map(pld)
    cds["aa_seq_len"] = cds["base"].map(slen)
    print(f"metapredict done   : {len(dis)} proteins")

    ok = cds.dropna(subset=["mean_plddt"])
    print(f"\n--- canonical property landscape (n={len(ok)}) ---")
    for c in ["mean_disorder", "mean_plddt"]:
        print(f"  {c:14s} mean {ok[c].mean():6.3f}  median {ok[c].median():6.3f}"
              f"  SD {ok[c].std():6.3f}  range {ok[c].min():.3f}-{ok[c].max():.3f}")
    cds.to_csv(OUT, index=False)
    print(f"\nwritten: {OUT}")


if __name__ == "__main__":
    main()
