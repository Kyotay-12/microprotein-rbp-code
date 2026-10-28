"""Export the 7,264 GENCODE Phase I Ribo-seq ORFs (S2 + S3) to a FASTA file.

Source: data/Mudge2022_SupplementaryTables.xlsx
  - sheet "S2. PHASE I Ribo-seq ORFs"   (replicated, n=3,085)
  - sheet "S3. Single-study Ribo-seq ORFs" (n=4,179)

FASTA record id = orf_name. Trailing stop-codon "*" is stripped from the
amino-acid sequence. Stops on any duplicate orf_name or empty/invalid sequence
(without substituting missing sequences).
"""
import sys
from pathlib import Path
import openpyxl

ROOT = Path(__file__).resolve().parent.parent
XLSX = ROOT / "data" / "Mudge2022_SupplementaryTables.xlsx"
OUT = ROOT / "data" / "orfs_7264.fasta"
SHEETS = ["S2. PHASE I Ribo-seq ORFs", "S3. Single-study Ribo-seq ORFs"]

VALID_AA = set("ACDEFGHIKLMNPQRSTVWY")


def load_orfs():
    wb = openpyxl.load_workbook(XLSX, read_only=True, data_only=True)
    records = []  # (orf_name, sheet, transcript, seq_no_stop)
    for sheet in SHEETS:
        ws = wb[sheet]
        rows = ws.iter_rows(values_only=True)
        header = next(rows)
        idx = {h: i for i, h in enumerate(header) if h}
        for col in ("orf_name", "orf_sequence", "transcript"):
            if col not in idx:
                raise ValueError(f"{sheet}: missing required column {col!r}")
        for row in rows:
            name = row[idx["orf_name"]]
            if name is None or not str(name).strip():
                continue
            seq = row[idx["orf_sequence"]]
            tx = row[idx["transcript"]]
            if not seq:
                raise ValueError(f"{name} ({sheet}): empty orf_sequence")
            seq = str(seq).strip()
            if seq.endswith("*"):
                seq = seq[:-1]
            bad = set(seq) - VALID_AA
            if bad:
                raise ValueError(f"{name} ({sheet}): non-standard residues {bad} in sequence")
            records.append((str(name), sheet, str(tx) if tx else "", seq))
    return records


def main():
    records = load_orfs()
    names = [r[0] for r in records]
    dupes = {n for n in names if names.count(n) > 1}
    if dupes:
        raise ValueError(f"Duplicate orf_name across sheets: {sorted(dupes)[:10]} ...")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w") as fh:
        for name, _sheet, _tx, seq in records:
            fh.write(f">{name}\n{seq}\n")
    print(f"Wrote {len(records)} ORFs to {OUT}")
    print(f"  source: {XLSX.name}  sheets: {SHEETS}")


if __name__ == "__main__":
    main()
