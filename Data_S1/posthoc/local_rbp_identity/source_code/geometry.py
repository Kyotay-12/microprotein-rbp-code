"""Real-data geometry for 100-nt windows at the Mudge ORF start.

All genomic and mature-transcript intervals returned are zero-based, half-open.
Transcript coordinates run 5-prime to 3-prime on both genomic strands. The
fixed windows are U=[t_start-100,t_start), D=[t_start,t_start+100), with the
ORF start as the common anchor. This module does not read any outcome signal,
fit models, or write files.
"""
from __future__ import annotations

from bisect import bisect_right
from collections import defaultdict
from hashlib import sha256
import gzip
import importlib.util
from pathlib import Path
import re

import pandas as pd


WINDOW_NT = 100  # User-prespecified fixed window length.
CATALOGUE_SHEET = "S2. PHASE I Ribo-seq ORFs"
_TX = re.compile(r'transcript_id "([^"]+)"')
_GENE = re.compile(r'gene_id "([^"]+)"')
_EXON = re.compile(r'exon_number (?:")?(\d+)(?:")?;')


def _digest(path):
    h = sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _source_files(root):
    root = Path(root)
    return {
        "properties": root / "analysis/results/step0_properties.csv",
        "catalogue": root / "analysis/data/Mudge2022_SupplementaryTables.xlsx",
        "gtf": root / "analysis/data/gencode.v35.annotation.gtf.gz",
        "id_helper": Path(__file__).resolve().with_name("ensembl_ids.py"),
    }


def _load_stable(path):
    spec = importlib.util.spec_from_file_location("local_feasibility_ensembl_ids", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.stable_ensembl_id


def _union(intervals):
    result = []
    for start, end in sorted(set(intervals)):
        if result and start <= result[-1][1]:
            result[-1] = (result[-1][0], max(end, result[-1][1]))
        else:
            result.append((start, end))
    return result


def _overlap(blocks, union):
    ends = [end for _, end in union]
    result = 0
    for start, end in blocks:
        index = bisect_right(ends, start)
        while index < len(union) and union[index][0] < end:
            a, b = union[index]
            result += max(0, min(end, b) - max(start, a))
            index += 1
    return result


def _integer(value):
    """Allow workbook integer cells or integer strings, never round values."""
    text = str(value).strip()
    if not re.fullmatch(r"\d+(?:\.0+)?", text):
        raise ValueError(f"not a nonnegative integer: {value!r}")
    return int(text.split(".", 1)[0])


def _catalogue_blocks(row):
    starts = [_integer(value) - 1 for value in re.split(r"[;,]", str(row["starts"]))]
    ends = [_integer(value) for value in re.split(r"[;,]", str(row["ends"]))]
    if len(starts) != len(ends) or not starts:
        raise ValueError("start/end count mismatch")
    blocks = sorted(zip(starts, ends))
    if any(start < 0 or end <= start for start, end in blocks):
        raise ValueError("nonpositive or out-of-range Mudge block")
    if any(a[1] > b[0] for a, b in zip(blocks, blocks[1:])):
        raise ValueError("overlapping Mudge blocks")
    return blocks


def _load_gtf(path, targets, stable):
    raw = {key: {"ids": set(), "chromosomes": set(), "strands": set(),
                 "genes": set(), "exons": [], "CDS": []} for key in targets}
    global_cds = defaultdict(list)
    with gzip.open(path, "rt") as handle:
        for line_number, line in enumerate(handle, 1):
            if line.startswith("#"):
                continue
            fields = line.rstrip("\n").split("\t")
            if len(fields) != 9:
                raise ValueError(f"Malformed GTF row {line_number}")
            feature = fields[2]
            if feature not in {"exon", "transcript", "CDS"}:
                continue
            chromosome, strand = fields[0], fields[6]
            start, end = int(fields[3]) - 1, int(fields[4])
            if start < 0 or end <= start or strand not in {"+", "-"}:
                raise ValueError(f"Invalid GTF interval at row {line_number}")
            if feature == "CDS":
                global_cds[(chromosome, strand)].append((start, end))
            match = _TX.search(fields[8])
            if not match:
                raise ValueError(f"Missing GTF transcript_id at row {line_number}")
            key = stable(match.group(1))
            if key not in targets:
                continue
            gene = _GENE.search(fields[8])
            if not gene:
                raise ValueError(f"Missing GTF gene_id at row {line_number}")
            record = raw[key]
            record["ids"].add(match.group(1))
            record["chromosomes"].add(chromosome)
            record["strands"].add(strand)
            record["genes"].add(stable(gene.group(1)))
            if feature == "exon":
                exon = _EXON.search(fields[8])
                record["exons"].append((start, end, int(exon.group(1)) if exon else None))
            elif feature == "CDS":
                record["CDS"].append((start, end))

    # No first match or union is allowed to resolve an ambiguous transcript key.
    ambiguous = sorted(key for key, record in raw.items()
                       if any(len(record[field]) > 1 for field in
                              ["ids", "chromosomes", "strands", "genes"]))
    if ambiguous:
        raise ValueError("Ambiguous PAR-preserving GTF transcript keys: " + ", ".join(ambiguous))
    transcripts, failures = {}, {}
    for key, record in raw.items():
        if not record["exons"]:
            failures[key] = "missing_gtf_exons"
            continue
        strand = next(iter(record["strands"]))
        genomic = sorted(record["exons"])
        if any(a[1] > b[0] for a, b in zip(genomic, genomic[1:])):
            failures[key] = "overlapping_or_duplicate_gtf_exons"
            continue
        ordered = sorted(record["exons"], reverse=(strand == "-"))
        if [entry[2] for entry in ordered] != list(range(1, len(ordered) + 1)):
            failures[key] = "gtf_exon_number_order_mismatch"
            continue
        offset = 0
        exons = []
        for start, end, exon_number in ordered:
            exons.append(dict(start=start, end=end, t_start=offset,
                              t_end=offset + end - start, exon_number=exon_number))
            offset += end - start
        transcripts[key] = dict(
            gtf_transcript_id=next(iter(record["ids"])),
            chromosome=next(iter(record["chromosomes"])), strand=strand,
            ensg=next(iter(record["genes"])), tx_length=offset, exons=exons,
            CDS=_union(record["CDS"]),
        )
    return transcripts, failures, {key: _union(blocks) for key, blocks in global_cds.items()}


def _genomic_to_transcript(blocks, transcript):
    path = []
    covered = 0
    for start, end in blocks:
        for exon in transcript["exons"]:
            lo, hi = max(start, exon["start"]), min(end, exon["end"])
            if lo >= hi:
                continue
            covered += hi - lo
            if transcript["strand"] == "+":
                path.append((exon["t_start"] + lo - exon["start"],
                             exon["t_start"] + hi - exon["start"]))
            else:
                path.append((exon["t_start"] + exon["end"] - hi,
                             exon["t_start"] + exon["end"] - lo))
    return _union(path), covered


def _transcript_to_genomic(start, end, transcript):
    if not 0 <= start <= end <= transcript["tx_length"]:
        raise ValueError("Transcript interval outside mature RNA")
    result = []
    for exon in transcript["exons"]:
        lo, hi = max(start, exon["t_start"]), min(end, exon["t_end"])
        if lo >= hi:
            continue
        if transcript["strand"] == "+":
            result.append([exon["start"] + lo - exon["t_start"],
                           exon["start"] + hi - exon["t_start"]])
        else:
            result.append([exon["end"] - hi + exon["t_start"],
                           exon["end"] - lo + exon["t_start"]])
    return result


def _validate_window_bases(start, end, blocks, transcript):
    """Independent per-base check of only real, observed <=100-nt windows."""
    expected = []
    offset = 0
    for exon in transcript["exons"]:
        exon_length = exon["end"] - exon["start"]
        for position in range(max(start, offset), min(end, offset + exon_length)):
            index = position - offset
            expected.append(exon["start"] + index if transcript["strand"] == "+"
                            else exon["end"] - 1 - index)
        offset += exon_length
    actual = []
    for lo, hi in blocks:
        actual.extend(range(lo, hi) if transcript["strand"] == "+"
                      else range(hi - 1, lo - 1, -1))
    if expected != actual or len(actual) != end - start:
        raise AssertionError("Real-data window base sequence failed independent validation")


def build_geometry(root, main_rows):
    """Return (per-ORF DataFrame, audit) without file writes.

    main_rows must have unique orf_name and original transcript; a supplied
    enst column must equal the corrected helper's PAR-preserving stable ID.
    Inputs must be a subset of the authoritative S2 <=100-aa main population.
    Invalid row geometry is retained with a reason. Ambiguous GTF keys and
    duplicated source keys fail the entire call, rather than choosing a row.
    Blocks in result rows are genomic [start,end] pairs in transcript order.
    audit['window_blocks'] is a flat query list for available windows only.
    """
    files = _source_files(root)
    before = {key: {"path": str(path), "sha256": _digest(path)}
              for key, path in files.items()}
    stable = _load_stable(files["id_helper"])
    required = {"orf_name", "transcript"}
    if not required.issubset(main_rows.columns):
        raise ValueError(f"main_rows missing columns: {sorted(required - set(main_rows.columns))}")
    if main_rows.orf_name.isna().any() or not main_rows.orf_name.is_unique:
        raise ValueError("main_rows must have unique nonmissing orf_name")
    properties = pd.read_csv(files["properties"], comment="#")
    if not properties.orf_name.is_unique:
        raise ValueError("Duplicate authoritative property-table ORF keys")
    population = properties.loc[properties.source_sheet.eq("S2") & properties.aa_length.le(100)]
    if not set(main_rows.orf_name).issubset(set(population.orf_name)):
        raise ValueError("Input ORFs outside authoritative S2 <=100-aa population")
    authoritative = population.set_index("orf_name")
    for row in main_rows.to_dict("records"):
        if row["transcript"] != authoritative.at[row["orf_name"], "transcript"]:
            raise ValueError(f"Original transcript changed from properties: {row['orf_name']}")
        if "enst" in row and row["enst"] != stable(row["transcript"]):
            raise ValueError(f"Input enst disagrees with PAR-preserving policy: {row['orf_name']}")
    catalogue = pd.read_excel(files["catalogue"], sheet_name=CATALOGUE_SHEET)
    fields = {"orf_name", "transcript", "gene_id", "chrm", "strand", "starts", "ends", "orf_length"}
    if not fields.issubset(catalogue.columns) or not catalogue.orf_name.is_unique:
        raise ValueError("Mudge catalogue schema or unique ORF key failed")
    catalogue = catalogue.set_index("orf_name")
    targets = {stable(value) for value in main_rows.transcript}
    transcripts, failures, global_cds = _load_gtf(files["gtf"], targets, stable)
    rows, window_blocks, examples = [], [], []
    validated_windows = 0
    for source in main_rows.to_dict("records"):
        orf_id, enst = source["orf_name"], stable(source["transcript"])
        output = dict(orf_name=orf_id, enst=enst, original_transcript=source["transcript"],
                      catalog_transcript=None, gtf_transcript_id=None, ensg=None,
                      chromosome=None, strand=None, geometry_valid=False,
                      geometry_reason="", t_start=None, t_end=None, tx_length=None,
                      orf_length_nt=None, orf_blocks=None, orf_tx_blocks=None,
                      U_available_nt=None, D_available_nt=None, U_full=False,
                      D_full=False, full_windows=False, window_reason="geometry_invalid",
                      U_start_tx=None, U_end_tx=None, D_start_tx=None, D_end_tx=None,
                      U_available_start_tx=None, U_available_end_tx=None,
                      D_available_start_tx=None, D_available_end_tx=None,
                      U_blocks=None, D_blocks=None, own_CDS_overlap_nt=None,
                      any_CDS_overlap_nt=None, U_any_CDS_overlap_nt=None,
                      D_any_CDS_overlap_nt=None)
        rows.append(output)
        if orf_id not in catalogue.index:
            output["geometry_reason"] = "missing_mudge_catalogue_row"
            continue
        record = catalogue.loc[orf_id]
        output["catalog_transcript"] = record.transcript
        output["chromosome"] = str(record.chrm) if str(record.chrm).startswith("chr") else "chr" + str(record.chrm)
        output["strand"] = record.strand
        try:
            catalogue_enst, gene = stable(record.transcript), stable(record.gene_id)
            blocks = _catalogue_blocks(record)
            recorded_length = _integer(record.orf_length)
        except (TypeError, ValueError) as error:
            output["geometry_reason"] = "invalid_mudge_record:" + str(error)
            continue
        output["ensg"] = gene
        output["orf_blocks"] = [list(block) for block in sorted(blocks, reverse=(record.strand == "-"))]
        length = sum(end - start for start, end in blocks)
        output["orf_length_nt"] = length
        if catalogue_enst != enst:
            output["geometry_reason"] = "mudge_property_transcript_mismatch"
            continue
        if length != recorded_length:
            output["geometry_reason"] = "mudge_block_length_disagrees_with_orf_length"
            continue
        if enst in failures:
            output["geometry_reason"] = failures[enst]
            continue
        transcript = transcripts[enst]
        output["gtf_transcript_id"] = transcript["gtf_transcript_id"]
        output["tx_length"] = transcript["tx_length"]
        if any(output[key] != transcript[key] for key in ["chromosome", "strand", "ensg"]):
            output["geometry_reason"] = "mudge_gtf_chromosome_strand_or_gene_mismatch"
            continue
        path, covered = _genomic_to_transcript(blocks, transcript)
        output["orf_tx_blocks"] = [list(block) for block in path]
        if covered != length:
            output["geometry_reason"] = "orf_not_fully_contained_in_host_exons"
            continue
        if len(path) != 1 or path[0][1] - path[0][0] != length:
            output["geometry_reason"] = "orf_not_contiguous_on_mature_transcript"
            continue
        start, end = path[0]
        if _union(map(tuple, _transcript_to_genomic(start, end, transcript))) != _union(blocks):
            raise AssertionError(f"Real-data ORF genomic round-trip failed: {orf_id}")
        output.update(geometry_valid=True, t_start=start, t_end=end,
                      own_CDS_overlap_nt=_overlap(blocks, transcript["CDS"]),
                      any_CDS_overlap_nt=_overlap(blocks, global_cds.get((output["chromosome"], output["strand"]), [])))
        truncation = []
        for side, ideal_start, ideal_end in [("U", start - WINDOW_NT, start),
                                             ("D", start, start + WINDOW_NT)]:
            available_start = max(0, ideal_start)
            available_end = min(transcript["tx_length"], ideal_end)
            available = available_end - available_start
            side_blocks = _transcript_to_genomic(available_start, available_end, transcript)
            _validate_window_bases(available_start, available_end, side_blocks, transcript)
            validated_windows += 1
            output.update({f"{side}_start_tx": ideal_start, f"{side}_end_tx": ideal_end,
                           f"{side}_available_start_tx": available_start,
                           f"{side}_available_end_tx": available_end,
                           f"{side}_available_nt": available,
                           f"{side}_full": available == WINDOW_NT,
                           f"{side}_blocks": side_blocks,
                           f"{side}_any_CDS_overlap_nt": _overlap(side_blocks, global_cds.get((output["chromosome"], output["strand"]), []))})
            if available < WINDOW_NT:
                truncation.append(f"{side}_absent" if available == 0 else f"{side}_truncated")
            for lo, hi in side_blocks:
                window_blocks.append(dict(orf_id=orf_id, side=side, chromosome=output["chromosome"],
                                          strand=output["strand"], start=lo, end=hi))
        output["full_windows"] = output["U_full"] and output["D_full"]
        output["window_reason"] = ";".join(truncation)
        for category, condition in [
            ("plus", output["strand"] == "+"),
            ("minus", output["strand"] == "-"),
            ("orf_exon_junction", len(output["orf_blocks"]) > 1),
            ("U_exon_junction", len(output["U_blocks"]) > 1),
            ("D_exon_junction", len(output["D_blocks"]) > 1),
        ]:
            if condition and category not in {entry["category"] for entry in examples}:
                examples.append({"category": category, "orf_name": orf_id, "enst": enst,
                                 "strand": output["strand"], "t_start": start, "t_end": end,
                                 "orf_blocks": output["orf_blocks"],
                                 "U_blocks": output["U_blocks"], "D_blocks": output["D_blocks"]})
    result = pd.DataFrame(rows)
    if result.empty:
        raise ValueError("Empty main_rows is not a feasibility population")
    for key in ["t_start", "t_end", "tx_length", "orf_length_nt", "U_available_nt", "D_available_nt",
                "U_start_tx", "U_end_tx", "D_start_tx", "D_end_tx", "U_available_start_tx",
                "U_available_end_tx", "D_available_start_tx", "D_available_end_tx",
                "own_CDS_overlap_nt", "any_CDS_overlap_nt", "U_any_CDS_overlap_nt", "D_any_CDS_overlap_nt"]:
        result[key] = pd.array(result[key], dtype="Int64")
    if result.orf_name.tolist() != main_rows.orf_name.tolist():
        raise AssertionError("Geometry output changed row identity or order")
    if any(_digest(files[key]) != value["sha256"] for key, value in before.items()):
        raise RuntimeError("Source file changed during geometry reconstruction")
    audit = dict(
        input_files=before, catalogue_sheet=CATALOGUE_SHEET,
        coordinate_convention="0-based half-open genomic and 5prime-to-3prime mature transcript",
        window_definition={"U": "[ORF_start-100,ORF_start)", "D": "[ORF_start,ORF_start+100)", "length_nt": WINDOW_NT},
        transcript_key_policy="numeric version removed, _PAR_Y preserved; ambiguous GTF keys fail",
        CDS_overlap_policy="all GENCODE v35 CDS intervals on the same chromosome and strand; caution only",
        input_n=len(main_rows), geometry_valid_n=int(result.geometry_valid.sum()),
        full_windows_n=int(result.full_windows.sum()),
        geometry_reasons=result.loc[~result.geometry_valid, "geometry_reason"].value_counts().to_dict(),
        window_reasons=result.loc[result.geometry_valid, "window_reason"].value_counts().to_dict(),
        source_rows_unchanged=True, window_blocks=window_blocks, transcripts=transcripts,
        validation={"real_windows_checked_base_by_base": validated_windows,
                    "ORF_genomic_roundtrip_checked_n": int(result.geometry_valid.sum()),
                    "examples": examples, "synthetic_data_used": False},
    )
    return result, audit
