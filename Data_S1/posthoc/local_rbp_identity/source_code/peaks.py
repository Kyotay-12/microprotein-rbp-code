"""Read-only eCLIP extraction within supplied local transcript windows.

The project root is the directory containing ``analysis/``. Window geometry is
supplied by the caller: every block must already be an exon-only piece of its
ORF's transcript, in GRCh38, zero-based half-open coordinates. This module does
not infer transcript geometry, read RPF/outcomes, write files, or access a network.

``count_window_rbps(root, window_blocks)`` returns JSON-serializable records at
ORF x supplied side x cell grain, with distinct RBP labels and counts. Missing
sides are not converted to zeros. Multiple files for the same RBP contribute to
one label. Metadata, exact file inventory, byte sizes, all gzip contents, interval
schema, and coordinates are validated before a result is returned.

The manifest and actual file byte hashes are returned for the caller's input
freeze. Consistency within an invocation is checked; the caller owns comparison
with the frozen audit specification across invocations. No peak-score, signal,
q-value, or blacklist filter is added to the existing step1 definition.
"""

from __future__ import annotations

import csv
import gzip
import hashlib
import io
import json
import operator
import re
from bisect import bisect_right
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping


CELLS = ("K562", "HepG2")
MANIFEST_REL = Path("analysis/results/eclip_manifest.tsv")
SOURCE_SCRIPTS = (
    "analysis/scripts/build_eclip_manifest.py",
    "analysis/scripts/download_encode.py",
    "analysis/scripts/step1_eclip_intersect.py",
    "analysis/scripts/step1c_region_intersect.py",
)
REQUIRED_BLOCK_FIELDS = {"orf_id", "side", "chromosome", "strand", "start", "end"}


class PeakInputError(ValueError):
    """A missing, changed, or inconsistent scientific input prevents counting."""


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _json_sha256(value: Any) -> str:
    return _sha256_bytes(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                   ensure_ascii=False, allow_nan=False).encode())


def _require_file(path: Path) -> None:
    if not path.is_file():
        raise PeakInputError(f"Required source file is missing: {path}")


def _manifest(root: Path) -> tuple[list[dict], dict]:
    path = root / MANIFEST_REL
    _require_file(path)
    manifest_bytes = path.read_bytes()
    manifest_sha = _sha256_bytes(manifest_bytes)
    try:
        reader = csv.DictReader(io.StringIO(manifest_bytes.decode("utf-8")), delimiter="\t")
        expected_fields = ["cell_line", "rbp", "experiment", "file", "href", "file_size"]
        if reader.fieldnames != expected_fields:
            raise PeakInputError(f"Unexpected eCLIP manifest schema: {reader.fieldnames}")
        rows = list(reader)
    except UnicodeError as exc:
        raise PeakInputError(f"Manifest is not UTF-8: {path}") from exc
    if not rows:
        raise PeakInputError("The eCLIP manifest has no experiments.")
    files_seen, experiments_seen = set(), set()
    expected_paths = {cell: set() for cell in CELLS}
    for row in rows:
        if any(not isinstance(row.get(key), str) or not row[key].strip()
               for key in expected_fields) or None in row:
            raise PeakInputError(f"Missing or malformed manifest field: {row}")
        cell, rbp, experiment, accession = (row[key] for key in
                                             ("cell_line", "rbp", "experiment", "file"))
        if cell not in CELLS:
            raise PeakInputError(f"Unexpected cell line in manifest: {cell}")
        if not re.fullmatch(r"[A-Za-z0-9_.-]+", rbp):
            raise PeakInputError(f"Unsafe or malformed RBP label: {rbp!r}")
        if not re.fullmatch(r"ENCSR[A-Z0-9]{6}", experiment):
            raise PeakInputError(f"Invalid experiment accession: {experiment}")
        if not re.fullmatch(r"ENCFF[A-Z0-9]{6}", accession):
            raise PeakInputError(f"Invalid file accession: {accession}")
        expected_href = (f"https://www.encodeproject.org/files/{accession}/"
                         f"@@download/{accession}.bed.gz")
        if row["href"] != expected_href:
            raise PeakInputError(f"File accession and manifest download URL disagree: {row}")
        if accession in files_seen or experiment in experiments_seen:
            raise PeakInputError(f"Duplicate file or experiment in manifest: {row}")
        files_seen.add(accession)
        experiments_seen.add(experiment)
        try:
            declared_size = int(row["file_size"])
        except ValueError as exc:
            raise PeakInputError(f"Invalid file size in manifest: {row}") from exc
        if declared_size <= 0:
            raise PeakInputError(f"Nonpositive file size in manifest: {row}")
        rel = Path("analysis/data/eclip") / cell / f"{rbp}__{experiment}__{accession}.bed.gz"
        source = root / rel
        _require_file(source)
        if source.stat().st_size != declared_size:
            raise PeakInputError(f"Peak-file size disagrees with manifest: {source}")
        row["relative_path"] = str(rel)
        row["path"] = str(source)
        row["size_bytes"] = declared_size
        expected_paths[cell].add(source.name)
    for cell in CELLS:
        actual = {p.name for p in (root / "analysis/data/eclip" / cell).glob("*.bed.gz")}
        if actual != expected_paths[cell]:
            raise PeakInputError(f"Peak directory differs from manifest for {cell}: "
                                 f"missing={sorted(expected_paths[cell] - actual)}, "
                                 f"extra={sorted(actual - expected_paths[cell])}")
        if not actual:
            raise PeakInputError(f"No assayed peak files for {cell}")
    scripts = []
    for rel in SOURCE_SCRIPTS:
        source = root / rel
        _require_file(source)
        content = source.read_bytes()
        scripts.append({"relative_path": rel, "path": str(source),
                        "sha256": _sha256_bytes(content), "size_bytes": len(content)})
    rows.sort(key=lambda row: (row["cell_line"], row["rbp"], row["experiment"], row["file"]))
    metadata = {
        "manifest_path": str(path), "manifest_relative_path": str(MANIFEST_REL),
        "manifest_sha256": manifest_sha, "source_scripts": scripts,
        "manifest_fields": expected_fields,
        "selection_from_source_script": {
            "experiment_assay_title": "eCLIP", "experiment_status": "released",
            "assembly": "GRCh38", "file_format": "bed", "output_type": "peaks",
            "file_status": "released", "biological_replicates": [1, 2],
            "multiple_candidate_preference": "preferred_default, otherwise first candidate",
        },
        "metadata_limitation": (
            "The saved manifest does not contain raw ENCODE API records, an explicit "
            "IDR threshold, blacklist-processing metadata, or file_format_type. "
            "Selection criteria above are traced to the local manifest-building script; "
            "per-file peak_names are observed directly in the saved BED contents."
        ),
    }
    return rows, metadata


def _integer_coordinate(value: Any, field: str) -> int:
    if isinstance(value, bool):
        raise PeakInputError(f"Boolean {field} coordinate is invalid")
    try:
        return operator.index(value)
    except TypeError as exc:
        raise PeakInputError(f"{field} must be an integer, not {value!r}") from exc


def _window_groups(window_blocks: Iterable[Mapping[str, Any]]) -> tuple[dict, list[dict]]:
    groups, orf_context, normalized = defaultdict(list), {}, []
    for index, block in enumerate(window_blocks):
        if not isinstance(block, Mapping) or not REQUIRED_BLOCK_FIELDS <= block.keys():
            raise PeakInputError(f"Window block {index} is missing required fields")
        orf, side, chrom, strand = (block[key] for key in
                                    ("orf_id", "side", "chromosome", "strand"))
        if not isinstance(orf, str) or not orf.strip():
            raise PeakInputError(f"Window block {index} has no valid ORF identifier")
        if side not in ("U", "D") or strand not in ("+", "-"):
            raise PeakInputError(f"Invalid window side or strand at block {index}")
        if not isinstance(chrom, str) or not chrom.startswith("chr") or len(chrom) < 4:
            raise PeakInputError(f"Window chromosome must use the source chr naming: {chrom!r}")
        start = _integer_coordinate(block["start"], "start")
        end = _integer_coordinate(block["end"], "end")
        if start < 0 or end <= start:
            raise PeakInputError(f"Invalid half-open window interval: {block}")
        if orf in orf_context and orf_context[orf] != (chrom, strand):
            raise PeakInputError(f"An ORF has conflicting chromosome/strand: {orf}")
        orf_context[orf] = (chrom, strand)
        groups[(orf, side)].append((chrom, strand, start, end))
        normalized.append({"orf_id": orf, "side": side, "chromosome": chrom,
                           "strand": strand, "start": start, "end": end})
    if not groups:
        raise PeakInputError("No nonempty window blocks supplied; no zero counts inferred")
    for key, blocks in groups.items():
        blocks.sort(key=lambda b: b[2:])
        for previous, current in zip(blocks, blocks[1:]):
            if previous[3] > current[2]:
                raise PeakInputError(f"Duplicate or overlapping exon-window blocks for {key}")
    normalized.sort(key=lambda b: tuple(b[key] for key in
                                       ("orf_id", "side", "chromosome", "strand", "start", "end")))
    return dict(groups), normalized


def _load_peak_intervals(row: dict) -> tuple[dict, dict]:
    path = Path(row["path"])
    _require_file(path)
    compressed = path.read_bytes()
    if len(compressed) != row["size_bytes"]:
        raise PeakInputError(f"Peak-file size changed while counting: {path}")
    try:
        text = gzip.decompress(compressed).decode("utf-8")
    except (OSError, EOFError, UnicodeError) as exc:
        raise PeakInputError(f"Unreadable or corrupt complete peak file: {path}") from exc
    by_locus, names, count = defaultdict(list), set(), 0
    strands, chromosomes = Counter(), Counter()
    for line_number, line in enumerate(text.splitlines(), 1):
        fields = line.split("\t")
        if len(fields) != 10:
            raise PeakInputError(f"Expected 10 BED narrowPeak fields at {path}:{line_number}")
        chrom, start_text, end_text, name, _, strand = fields[:6]
        try:
            start, end = int(start_text), int(end_text)
        except ValueError as exc:
            raise PeakInputError(f"Noninteger peak coordinate at {path}:{line_number}") from exc
        if not chrom.startswith("chr") or strand not in ("+", "-") or start < 0 or end <= start:
            raise PeakInputError(f"Invalid peak locus at {path}:{line_number}")
        by_locus[(chrom, strand)].append((start, end))
        names.add(name)
        strands[strand] += 1
        chromosomes[chrom] += 1
        count += 1
    if not count:
        raise PeakInputError(f"Empty peak file is not interpreted as no binding: {path}")
    metadata = {key: row[key] for key in
                ("cell_line", "rbp", "experiment", "file", "href", "relative_path", "path", "size_bytes")}
    metadata.update({"sha256": _sha256_bytes(compressed), "n_peaks": count,
                     "peak_names": sorted(names), "all_peak_names_end_with_IDR":
                         all(name.endswith("_IDR") for name in names),
                     "peaks_by_strand": dict(sorted(strands.items())),
                     "peaks_by_chromosome": dict(sorted(chromosomes.items()))})
    return dict(by_locus), metadata


def _merged_index(intervals: list[tuple[int, int]]) -> tuple[list[int], list[int]]:
    """Union coverage within one file and locus; merge adjacency without adding nt."""
    starts, ends = [], []
    for start, end in sorted(intervals):
        if starts and start <= ends[-1]:
            ends[-1] = max(end, ends[-1])
        else:
            starts.append(start)
            ends.append(end)
    return starts, ends


def _overlaps(index: tuple[list[int], list[int]], start: int, end: int) -> bool:
    starts, ends = index
    # A merged interval must end strictly after the block starts and start
    # strictly before the block ends. Exact touching therefore is not a hit.
    candidate = bisect_right(ends, start)
    return candidate < len(starts) and starts[candidate] < end


def count_window_rbps(root: str | Path, window_blocks: Iterable[Mapping[str, Any]],
                      *, validate: bool = True, validation_windows: int = 24) -> dict:
    """Count distinct assayed RBPs at each supplied ORF/side in each cell.

    ``records`` has keys orf_id, side, cell_line, nrbp, rbps, n_blocks,
    window_nt. ``rbps`` and ``available_rbps[cell]`` are sorted unique lists,
    i.e. JSON-serializable set representations. A supplied valid window with no
    overlaps returns nrbp=0/rbps=[] only after all peak-file checks pass. Missing
    windows are absent from records and listed in diagnostics, never zero-filled.

    By default a deterministic subset of real supplied windows is checked with
    an independent direct interval scan over every source file. Validation uses
    no synthetic windows, mock peaks, or outcome values.
    """
    root = Path(root).resolve()
    rows, provenance = _manifest(root)
    groups, normalized = _window_groups(window_blocks)
    labels = {(orf, side, cell): set() for orf, side in groups for cell in CELLS}
    available = {cell: sorted({row["rbp"] for row in rows if row["cell_line"] == cell})
                 for cell in CELLS}
    by_locus = defaultdict(list)
    for key, blocks in groups.items():
        for chrom, strand, start, end in blocks:
            by_locus[(chrom, strand)].append((key, start, end))
    file_metadata, covered_loci = [], set()
    for row in rows:
        peaks, metadata = _load_peak_intervals(row)
        covered_loci.update(peaks)
        file_metadata.append(metadata)
        for locus in by_locus.keys() & peaks.keys():
            index = _merged_index(peaks[locus])
            for (orf, side), start, end in by_locus[locus]:
                key = (orf, side, row["cell_line"])
                if row["rbp"] not in labels[key] and _overlaps(index, start, end):
                    labels[key].add(row["rbp"])
    # Every coordinate label must occur somewhere in the source assembly data.
    # An absent strand/chromosome pair can be a genuine zero, but an entirely
    # unknown chromosome label is a coordinate-system problem, not a zero.
    peak_chromosomes = {chrom for chrom, _ in covered_loci}
    unknown_chromosomes = sorted({block["chromosome"] for block in normalized} - peak_chromosomes)
    if unknown_chromosomes:
        raise PeakInputError(f"Window chromosomes absent from all peak sources: {unknown_chromosomes}")
    records = []
    for orf, side in sorted(groups):
        blocks = groups[(orf, side)]
        for cell in CELLS:
            rbps = sorted(labels[(orf, side, cell)])
            records.append({"orf_id": orf, "side": side, "cell_line": cell,
                            "nrbp": len(rbps), "rbps": rbps, "n_blocks": len(blocks),
                            "window_nt": sum(end - start for _, _, start, end in blocks)})
    if _sha256_bytes((root / MANIFEST_REL).read_bytes()) != provenance["manifest_sha256"]:
        raise PeakInputError("Manifest changed during extraction")
    for script in provenance["source_scripts"]:
        if _sha256_bytes(Path(script["path"]).read_bytes()) != script["sha256"]:
            raise PeakInputError(f"Source script changed during extraction: {script['path']}")
    source_inventory = [{key: row[key] for key in
                         ("cell_line", "rbp", "experiment", "file", "relative_path", "size_bytes", "sha256")}
                        for row in file_metadata]
    provenance.update({
        "root": str(root), "peak_files": file_metadata,
        "peak_inventory_sha256": _json_sha256(source_inventory),
        "window_blocks_sha256": _json_sha256(normalized),
        "records_sha256": _json_sha256(records),
        "module_sha256": _sha256_bytes(Path(__file__).read_bytes()),
        "filtering": {
            "coordinate_system": "GRCh38, zero-based half-open",
            "region": "caller-supplied exon-only transcript-window blocks",
            "strand": "same", "minimum_overlap_nt": 1,
            "count_unit": "distinct RBP per ORF-side-cell, union across all assigned files",
            "additional_score_signal_p_q_filters": "none",
            "additional_blacklist_filter": "none; matches existing step1 and region intersection scripts",
            "upstream_blacklist_handling": "not established from locally saved metadata",
            "peak_interval_merging": "within each file and chromosome/strand, union preserves coverage",
        },
    })
    orfs = sorted({orf for orf, _ in groups})
    result = {
        "schema_version": "local_rbp_window_counts.v1", "cells": list(CELLS),
        "available_rbps": available,
        "shared_assayed_rbps": sorted(set(available[CELLS[0]]) & set(available[CELLS[1]])),
        "records": records, "provenance": provenance,
        "diagnostics": {
            "n_window_blocks": len(normalized), "n_orfs": len(orfs),
            "n_supplied_windows": len(groups), "n_output_records": len(records),
            "missing_sides": [{"orf_id": orf, "side": side} for orf in orfs for side in ("U", "D")
                              if (orf, side) not in groups],
            "peak_files_by_cell": dict(Counter(row["cell_line"] for row in rows)),
            "peak_intervals_by_cell": {cell: sum(row["n_peaks"] for row in file_metadata
                                                  if row["cell_line"] == cell) for cell in CELLS},
            "assayed_rbps_by_cell": {cell: len(available[cell]) for cell in CELLS},
            "repeated_rbp_experiments": {cell: {rbp: count for rbp, count in
                sorted(Counter(row["rbp"] for row in rows if row["cell_line"] == cell).items())
                if count > 1} for cell in CELLS},
            "all_files_peak_names_end_with_IDR": all(row["all_peak_names_end_with_IDR"]
                                                       for row in file_metadata),
            "true_zero_window_records_by_cell": {cell: sum(row["nrbp"] == 0 for row in records
                                                            if row["cell_line"] == cell) for cell in CELLS},
        },
    }
    result["validation"] = (validate_window_subset(root, normalized, result,
                                                   max_windows=validation_windows)
                            if validate else {"performed": False})
    return result


def validate_window_subset(root: str | Path, window_blocks: Iterable[Mapping[str, Any]],
                           result: Mapping[str, Any], *, max_windows: int = 24) -> dict:
    """Independently compare source-file brute-force overlaps on real windows.

    Sampling covers available strand/side, single/multiple block, zero/nonzero
    combinations, then fills lexically. This is an implementation validation,
    not a statistical inference or a representativeness claim.
    """
    if isinstance(max_windows, bool) or not isinstance(max_windows, int) or max_windows < 1:
        raise PeakInputError("max_windows must be a positive integer")
    root = Path(root).resolve()
    rows, manifest_provenance = _manifest(root)
    groups, normalized = _window_groups(window_blocks)
    if _json_sha256(normalized) != result["provenance"]["window_blocks_sha256"]:
        raise PeakInputError("Validation windows differ from those used for extraction")
    if manifest_provenance["manifest_sha256"] != result["provenance"]["manifest_sha256"]:
        raise PeakInputError("Manifest changed between extraction and independent validation")
    if _json_sha256(result["records"]) != result["provenance"]["records_sha256"]:
        raise PeakInputError("Output records changed before independent validation")
    lookup = {(r["orf_id"], r["side"], r["cell_line"]): r for r in result["records"]}
    expected_keys = {(*key, cell) for key in groups for cell in CELLS}
    if len(lookup) != len(result["records"]) or set(lookup) != expected_keys:
        raise PeakInputError("Extraction record grain is not ORF-side-cell")
    chosen, seen_strata = [], set()
    for key in sorted(groups):
        blocks = groups[key]
        strata = {(blocks[0][1], key[1], len(blocks) > 1, cell,
                   lookup[(*key, cell)]["nrbp"] > 0) for cell in CELLS}
        if strata - seen_strata:
            chosen.append(key)
            seen_strata.update(strata)
        if len(chosen) >= max_windows:
            break
    chosen_set = set(chosen)
    for key in sorted(groups):
        if len(chosen) >= max_windows:
            break
        if key not in chosen_set:
            chosen.append(key)
            chosen_set.add(key)
    direct_hits = {(*key, cell): set() for key in chosen for cell in CELLS}
    checks_by_locus = defaultdict(list)
    for key in chosen:
        for chrom, strand, start, end in groups[key]:
            checks_by_locus[(chrom, strand)].append((key, start, end))
    expected_hashes = {row["file"]: row["sha256"] for row in result["provenance"]["peak_files"]}
    n_read, n_pairs = 0, 0
    for row in rows:
        path = Path(row["path"])
        if _sha256_bytes(path.read_bytes()) != expected_hashes[row["file"]]:
            raise PeakInputError(f"Peak file changed before independent validation: {path}")
        # Independent parser and unmerged direct comparisons. Do not call
        # _load_peak_intervals, _merged_index, or _overlaps here.
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            for line in handle:
                fields = line.rstrip("\n").split("\t")
                chrom, strand = fields[0], fields[5]
                peak_start, peak_end = int(fields[1]), int(fields[2])
                n_read += 1
                for key, start, end in checks_by_locus.get((chrom, strand), ()):
                    n_pairs += 1
                    if min(peak_end, end) - max(peak_start, start) >= 1:
                        direct_hits[(*key, row["cell_line"])].add(row["rbp"])
    comparisons = []
    for key in chosen:
        for cell in CELLS:
            observed = lookup[(*key, cell)]
            expected = sorted(direct_hits[(*key, cell)])
            if expected != observed["rbps"] or len(expected) != observed["nrbp"]:
                raise PeakInputError(f"Independent direct scan disagrees for {(*key, cell)}: "
                                     f"direct={expected}, indexed={observed['rbps']}")
            comparisons.append({"orf_id": key[0], "side": key[1], "cell_line": cell,
                                "strand": groups[key][0][1], "n_blocks": len(groups[key]),
                                "nrbp_direct": len(expected), "rbps_direct": expected,
                                "matches": True})
    return {
        "performed": True, "method": "independent unmerged direct interval scan of every peak file",
        "data": "deterministic subset of actual supplied exon-window blocks; no synthetic input",
        "n_selected_windows": len(chosen), "n_cell_window_comparisons": len(comparisons),
        "n_source_files_read": len(rows), "n_source_peaks_read": n_read,
        "n_direct_interval_comparisons": n_pairs, "all_matched": True,
        "selected_windows_sha256": _json_sha256([list(key) for key in chosen]),
        "comparisons": comparisons,
    }
