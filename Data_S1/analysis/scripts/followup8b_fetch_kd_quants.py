"""Followup 8b — stream ENCODE KD/control transcript quantifications and keep
only the ncORF host transcripts.

Each manifest-listed file is streamed through curl and filtered in memory
to the target ncORF host transcripts; nothing is
written to disk except the filtered rows. Resumable: files already extracted
are skipped.

Output: data/kd/quant/<FILE_ACCESSION>.tsv  (enst_base, TPM)
Failures raise; no dummy values.
"""
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import subprocess
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
KD = ROOT / "data" / "kd"
MAN = KD / "kd_manifest.tsv"
OUTDIR = KD / "quant"
TARGETS = KD / "target_ensts.txt"
WORKERS = 5


def fetch_one(acc, href, targets):
    out = OUTDIR / f"{acc}.tsv"
    if out.exists() and out.stat().st_size > 0:
        return acc, "skip"
    p = subprocess.Popen(["curl", "-sfL", href], stdout=subprocess.PIPE)
    keep = []
    header = None
    for i, raw in enumerate(p.stdout):
        line = raw.decode("utf-8", "replace").rstrip("\n")
        if i == 0:
            header = line.split("\t")
            # ENCODE serves two quantification formats for this assay:
            #   RSEM     : transcript_id ... TPM
            #   kallisto : target_id length eff_length est_counts tpm
            # Both are accepted; the format is recorded so that followup8d can
            # refuse to compare a KD and a control quantified differently.
            if "transcript_id" in header and "TPM" in header:
                ti, tp, fmt = header.index("transcript_id"), header.index("TPM"), "rsem"
            elif "target_id" in header and "tpm" in header:
                ti, tp, fmt = header.index("target_id"), header.index("tpm"), "kallisto"
            else:
                p.kill()
                raise RuntimeError(f"{acc}: unexpected header {header[:5]}")
            continue
        f = line.split("\t")
        if len(f) <= tp:
            continue
        # kallisto target_id can be a full GENCODE header (ENST|ENSG|...)
        base = f[ti].split("|")[0].split(".")[0]
        if base in targets:
            keep.append((base, f[tp]))
    p.stdout.close()
    if p.wait() != 0:
        raise RuntimeError(f"{acc}: curl failed")
    if not keep:
        raise RuntimeError(f"{acc}: no target transcripts found")
    df = pd.DataFrame(keep, columns=["enst", "TPM"])
    df["TPM"] = df["TPM"].astype(float)
    df = df.groupby("enst", as_index=False)["TPM"].sum()
    df["format"] = fmt
    df.to_csv(out, sep="\t", index=False)
    return acc, f"ok n={len(df)} fmt={fmt}"


def main():
    targets = set(TARGETS.read_text().split())
    man = pd.read_csv(MAN, sep="\t").drop_duplicates("file")
    OUTDIR.mkdir(parents=True, exist_ok=True)
    todo = [(r.file, r.href) for r in man.itertuples()]
    print(f"files: {len(todo)}  targets: {len(targets)}  workers: {WORKERS}", flush=True)
    done = 0
    with ThreadPoolExecutor(WORKERS) as ex:
        futs = {ex.submit(fetch_one, a, h, targets): a for a, h in todo}
        for fu in as_completed(futs):
            try:
                acc, msg = fu.result()
            except Exception as e:
                print(f"  FAILED: {e}", flush=True)
                raise
            done += 1
            if done % 25 == 0:
                print(f"  {done}/{len(todo)}", flush=True)
    print(f"done: {done} files -> {OUTDIR}")


if __name__ == "__main__":
    main()
