"""Plot canonical and ncORF reference results without refitting models.

Input files and plotted records are recorded in sourceclaims.
"""
from pathlib import Path
import argparse
import ast
import hashlib
import json
import os

HERE = Path(__file__).resolve().parent
import numpy as np
import pandas as pd
from PIL import Image

PARSER = argparse.ArgumentParser()
PARSER.add_argument("--project-root", type=Path, required=True)
PARSER.add_argument("--canonical-models", type=Path)
PARSER.add_argument("--outdir", type=Path, required=True)
ARGS = PARSER.parse_args()
ROOT = ARGS.project_root.resolve()
OUT = ARGS.outdir.resolve()
CANONICAL = (ARGS.canonical_models or ROOT / "posthoc/canonical/results/canonical_models.csv").resolve()
RES = ROOT / "analysis" / "results"
SOURCE = HERE.parent / "analysis" / "scripts" / "make_figures.py"
for path in [CANONICAL, SOURCE, *[RES / n for n in ['followup1_partial_r2.csv','followup3_per_rbp.csv','followup4_gc_vs_property.csv']]]:
    if not path.is_file():raise FileNotFoundError(f'Required separate result or code: {path}')
for protected in [ROOT, HERE.parent.parent]:
    if OUT==protected or OUT.is_relative_to(protected) or protected.is_relative_to(OUT):
        raise ValueError('Output must be separate from code and input directories.')
if OUT.exists():raise FileExistsError('Output already exists; choose a new directory.')
OUT.mkdir(parents=True)
os.environ.setdefault("MPLCONFIGDIR", str(OUT / "matplotlib_cache"))
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
STEM = "figure4"
SOURCES = {}
READS = {}
CLAIMS = {"A": [], "B": [], "C": [], "D": []}
FIG_W_MM = 173.0  # Requested final publication width, not an empirical value.
FIG_W = FIG_W_MM / 25.4
K_COL = "#3266cc"
H_COL = "#e67e22"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def clean(obj):
    if isinstance(obj, dict):
        return {str(k): clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [clean(v) for v in obj]
    if isinstance(obj, np.generic):
        return clean(obj.item())
    if isinstance(obj, float) and not np.isfinite(obj):
        return None
    return obj


def read_result(name, **kwargs):
    path = CANONICAL if name == "canonical_models.csv" else RES / name
    if not path.is_file():
        raise FileNotFoundError(path)
    SOURCES[str(path)] = sha(path)
    df = pd.read_csv(path, **kwargs)
    if name == "canonical_models.csv":
        df = df[(df.population == "all_phase_valid") & (df.family == "occupancy") &
                (df.outcome == "lrpf") &
                (df.specification == "all_tags__preserve_PAR__corrected_codon_length__phase_aware")]
        if len(df) != 2 or set(df.cell) != {"K562", "HepG2"}:
            raise ValueError("Canonical primary occupancy selector is not unique per cell")
    READS[name] = df.copy()
    return df


def unique_row(frame, selector):
    mask = pd.Series(True, index=frame.index)
    for key, value in selector.items():
        mask &= frame[key] == value
    selected = frame.loc[mask]
    if len(selected) != 1:
        raise ValueError(f"Expected one row for {selector}; got {len(selected)}")
    return selected.iloc[0]


def valid_ci(beta, lo, hi):
    values = np.array([beta, lo, hi], dtype=float)
    if not np.isfinite(values).all() or not lo <= beta <= hi:
        raise ValueError(f"Invalid coefficient/interval: {values}")


def record_values():
    r2 = READS["followup1_partial_r2.csv"]
    for analysis, outcome in [
        ("Step1_axis_a", "mean_plddt"),
        ("Step1_axis_a", "catgranule2_llps_score"),
        ("Step1_axis_a", "foldable"),
        ("Step1_axis_a", "llps_prone"),
        ("Step2_lRPF_main_TPMge1", "lRPF"),
        ("Step2_lRPF_lncRNA_TPMge1", "lRPF"),
    ]:
        for cell in ("K562", "HepG2"):
            selector = dict(analysis=analysis, outcome=outcome, cell=cell)
            row = unique_row(r2, selector)
            value = float(row.semi_partial_R2) * 100
            if not np.isfinite(value):
                raise ValueError(selector)
            CLAIMS["A"].append(dict(source="analysis/results/followup1_partial_r2.csv",
                                    selector=selector, record=row.to_dict(), plotted=value,
                                    transform="semi_partial_R2 * 100"))

    for name, population in [
        ("step2_tpm_sliding.csv", "ncORF main population"),
        ("canonical_models.csv", "canonical CDS reference"),
    ]:
        data = READS[name]
        for cell in ("K562", "HepG2"):
            selector = {"cell": cell}
            if name == "step2_tpm_sliding.csv":
                selector.update(stratum="main", tpm_cutoff=1.0)
            row = unique_row(data, selector)
            if name == "canonical_models.csv":
                selector.update(population="all_phase_valid", family="occupancy", outcome="lrpf",
                                specification="all_tags__preserve_PAR__corrected_codon_length__phase_aware")
            valid_ci(row.beta, row.ci_lo, row.ci_hi)
            CLAIMS["B"].append(dict(source="canonical_models" if name == "canonical_models.csv" else f"analysis/results/{name}", selector=selector,
                                    population=population, record=row.to_dict(),
                                    plotted=dict(beta=row.beta, ci_lo=row.ci_lo, ci_hi=row.ci_hi),
                                    transform="identity", ci_level=0.95,
                                    mature_host_length_adjustment=False))

    per_rbp = READS["followup3_per_rbp.csv"]
    selected = pd.concat([
        per_rbp[(per_rbp.cell == cell) & (per_rbp.beta > 0) & (per_rbp.p_bonf < 0.05)]
        .sort_values("beta", ascending=False).head(12)
        for cell in ("K562", "HepG2")
    ], ignore_index=True)
    rbp_order = selected.sort_values("beta", ascending=False).rbp.unique()[:14]
    for cell in ("K562", "HepG2"):
        for rbp in rbp_order:
            selector = dict(cell=cell, rbp=rbp)
            selected_row = per_rbp[(per_rbp.cell == cell) & (per_rbp.rbp == rbp)]
            if len(selected_row) == 0:
                CLAIMS["C"].append(dict(source="analysis/results/followup3_per_rbp.csv",
                                        selector=selector, plotted=False,
                                        reason="No source row; original panel leaves this position empty"))
                continue
            row = unique_row(per_rbp, selector)
            interval = np.array([row.beta, row.ci_lo, row.ci_hi], dtype=float)
            if np.isnan(interval).all():
                CLAIMS["C"].append(dict(source="analysis/results/followup3_per_rbp.csv",
                                        selector=selector, record=row.to_dict(), plotted=False,
                                        reason="Source coefficient and both confidence limits are undefined; original panel leaves this position empty"))
                continue
            valid_ci(row.beta, row.ci_lo, row.ci_hi)
            CLAIMS["C"].append(dict(source="analysis/results/followup3_per_rbp.csv",
                                    selector=selector, record=row.to_dict(),
                                    plotted=dict(beta=row.beta, ci_lo=row.ci_lo, ci_hi=row.ci_hi),
                                    transform="identity", ci_level=0.95))

    gc = READS["followup4_gc_vs_property.csv"]
    for cell in ("K562", "HepG2"):
        for outcome in ("mean_plddt", "catgranule2_llps_score", "foldable", "llps_prone"):
            selector = dict(cell=cell, outcome=outcome)
            row = unique_row(gc, selector)
            scale = 0.1 * (100 if outcome == "catgranule2_llps_score" else 1)
            beta, lo, hi = [float(row[col]) * scale for col in ("gc_beta", "ci_lo", "ci_hi")]
            valid_ci(beta, lo, hi)
            CLAIMS["D"].append(dict(source="analysis/results/followup4_gc_vs_property.csv",
                                    selector=selector, record=row.to_dict(),
                                    plotted=dict(beta=beta, ci_lo=lo, ci_hi=hi),
                                    scale=scale, transform="coefficient and interval * scale",
                                    ci_level=0.95))


def save(fig, stem):
    # Input/record validation takes place before any figure is saved.
    record_values()
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    canvas = fig.bbox
    outside = []
    for obj in fig.findobj(match=mpl.text.Text):
        if not obj.get_visible() or not obj.get_text():
            continue
        bounds = obj.get_window_extent(renderer)
        # Inactive alternate-side axis labels may be positioned outside canvas.
        if bounds.width == 0 or bounds.height == 0:
            continue
        if bounds.x0 < -0.75 or bounds.y0 < -0.75 or bounds.x1 > canvas.x1 + 0.75 or bounds.y1 > canvas.y1 + 0.75:
            outside.append(obj.get_text())
    if outside:
        raise ValueError(f"Text outside fixed canvas: {outside}")
    fig.savefig(OUT / f"{stem}.png", dpi=300)
    fig.savefig(OUT / f"{stem}.pdf", metadata={"Title": "Figure 4", "Author": "Kyota Yasuda"})
    with Image.open(OUT / f"{stem}.png") as im:
        size = im.size
        dpi = im.info.get("dpi")
    if not abs(size[0] / 300 * 25.4 - FIG_W_MM) < 0.1:
        raise ValueError("Saved raster width differs from requested 173 mm")
    for source_path, digest in SOURCES.items():
        if sha(Path(source_path)) != digest:
            raise ValueError(f"Input changed during build: {source_path}")
    payload = dict(
        title="Figure 4 corrected canonical-CDS adoption",
        generator="make_figure4_corrected.py", generator_sha256=sha(Path(__file__)),
        source_generator="analysis/scripts/make_figures.py", source_generator_sha256=sha(SOURCE),
        empirical_input_sha256={os.path.relpath(path, ROOT): digest for path, digest in SOURCES.items()},
        author_specified_geometry=dict(width_mm=FIG_W_MM, height_inches=6.4, png_dpi=300),
        changes=["Panel B canonical coefficients and intervals use all-tags, PAR-preserved, phase-aware corrected-codon-length reference results",
                 "Original ncORF coefficients retained without mature host-RNA length adjustment",
                 "No coefficient-ratio annotation or fold-difference headline",
                 "Panels A, C and D select the same records and apply the same transforms as the source figure"],
        model_note="Panel B compares different populations under the original model specification without mature host-RNA length; corrected canonical coding length does not add a mature host-RNA-length covariate.",
        plotted_records=CLAIMS,
        validation=dict(all_plotted_coefficients_finite_and_within_CI=True,
                        all_text_inside_fixed_canvas=True,
                        png_pixels=size, png_dpi_metadata=dpi),
        outputs={f"{stem}.{ext}": sha(OUT / f"{stem}.{ext}") for ext in ("png", "pdf")},
    )
    (OUT / "sourceclaims.json").write_text(json.dumps(clean(payload), indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    print(json.dumps(dict(output_directory=str(OUT), outputs=payload["outputs"], pixels=size)))


SOURCE_TEXT = SOURCE.read_text()
tree = ast.parse(SOURCE_TEXT)
function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "figure4")
function_text = ast.get_source_segment(SOURCE_TEXT, function)

def replace_once(old, new):
    global function_text
    if function_text.count(old) != 1:
        raise ValueError(f"Expected exactly one source fragment: {old!r}")
    function_text = function_text.replace(old, new, 1)

replace_once('read_result("followup10_canonical_lrpf_fixed.csv")',
             'read_result("canonical_models.csv")')
replace_once('"Canonical CDS reference"', '"Models without host-RNA length"')
ratio_loop = '''    for i, (n, c, chi) in enumerate(zip(nc_beta, can_beta, can_hi)):
        # clear the canonical error-bar cap, not just the bar top
        ax.text(i, chi + 0.07, f"nc/can = {n/c:.2f}",
                ha="center", fontsize=6.5, color="#222")'''
replace_once(ratio_loop, "    # Ratios are intentionally not displayed: populations differ.")

# Load figure style without running the source module's output actions.
style_node = next(node for node in tree.body if isinstance(node, ast.Expr)
                  and isinstance(node.value, ast.Call)
                  and ast.unparse(node.value.func) == "mpl.rcParams.update")
exec(compile(ast.Module(body=[style_node], type_ignores=[]), str(SOURCE), "exec"), globals())
mpl.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42, "savefig.bbox": None})
exec(compile(function_text, str(SOURCE), "exec"), globals())
figure4()
