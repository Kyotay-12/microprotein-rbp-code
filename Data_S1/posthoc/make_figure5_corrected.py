"""Plot already-validated result records; never refit or simulate data."""
from pathlib import Path
import argparse
import hashlib
import json
import math
import os
import sys

PARSER = argparse.ArgumentParser()
PARSER.add_argument("--project-root", type=Path, required=True)
PARSER.add_argument("--occupancy", type=Path)
PARSER.add_argument("--structural", type=Path)
PARSER.add_argument("--within-host", type=Path)
PARSER.add_argument("--structural-spec", type=Path)
PARSER.add_argument("--outdir", type=Path, required=True)
ARGS = PARSER.parse_args()
ROOT = ARGS.project_root.resolve()
OUT = ARGS.outdir.resolve()
SOURCES = {
    "occupancy": ARGS.occupancy or ROOT / "posthoc/occupancy/results.json",
    "structural": ARGS.structural or ROOT / "posthoc/structural/results.json",
    "within_host": ARGS.within_host or ROOT / "posthoc/within_host/results.json",
}
SPEC = ARGS.structural_spec or ROOT / "posthoc/structural/specification.json"
for path in [*SOURCES.values(), SPEC]:
    if not path.is_file():raise FileNotFoundError(f'Required separate result/specification: {path}')
for protected in [ROOT, Path(__file__).resolve().parents[2]]:
    if OUT==protected or OUT.is_relative_to(protected) or protected.is_relative_to(OUT):
        raise ValueError('Output must be separate from the code and input directories.')
if OUT.exists():raise FileExistsError('Output already exists; choose a new directory.')
OUT.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(OUT / "matplotlib_cache"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.text import Text

SOURCES = {
    "occupancy": ARGS.occupancy or ROOT / "posthoc/occupancy/results.json",
    "structural": ARGS.structural or ROOT / "posthoc/structural/results.json",
    "within_host": ARGS.within_host or ROOT / "posthoc/within_host/results.json",
}
SPEC = ARGS.structural_spec or ROOT / "posthoc/structural/specification.json"
RAW = {key: json.loads(path.read_text()) for key, path in SOURCES.items()}
BOUND = json.loads(SPEC.read_text())["bounds"]["primary"]
COLOR = {"K562": "#0072B2", "HepG2": "#E69F00"}
OFFSET = {"K562": 0.09, "HepG2": -0.09}
PLOTTED = []

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def record(source, **criteria):
    found = [r for r in RAW[source]["model_results"] if all(r.get(k) == v for k, v in criteria.items())]
    if len(found) != 1:
        raise ValueError((source, criteria, len(found)))
    return found[0]

def mark(ax, row, source, criteria, *, panel, standardized=False, ci=95):
    r = record(source, **criteria)
    beta = r["beta_std" if standardized else "beta"]
    lo, hi = r[f"ci{ci}_lo"], r[f"ci{ci}_hi"]
    if not all(math.isfinite(x) for x in (beta, lo, hi)) or not lo <= beta <= hi:
        raise ValueError((source, criteria, "invalid CI"))
    cell = r["cell"]
    ax.errorbar(beta, row + OFFSET[cell], xerr=[[beta-lo],[hi-beta]],
                fmt="o" if cell == "K562" else "s", markersize=4.1,
                color=COLOR[cell], ecolor=COLOR[cell], elinewidth=1.25,
                capsize=2.0, capthick=1.0, zorder=3)
    PLOTTED.append({"panel": panel, "source": source, "selection": criteria,
                    "x": beta, "ci_percent": ci, "ci_lo": lo, "ci_hi": hi,
                    "y_row": row, "record": r})
    return r

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 7.5,
    "axes.titlesize": 8.3, "axes.titleweight": "normal", "axes.labelsize": 7.4,
    "xtick.labelsize": 7.2, "ytick.labelsize": 7.3,
    "axes.linewidth": 0.7, "xtick.major.width": 0.7, "ytick.major.width": 0.7,
    "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.bbox": None,
    "axes.unicode_minus": True,
})
WIDTH_MM, HEIGHT_MM = 173.0, 157.0
fig, axs = plt.subplots(2, 2, figsize=(WIDTH_MM / 25.4, HEIGHT_MM / 25.4))
fig.subplots_adjust(left=0.16, right=0.975, bottom=0.11, top=0.83, wspace=0.66, hspace=1.0)
titles = ["Host-length sensitivity", "Length-adjusted properties", "Paired transcript occupancy", "Paired density ratio"]

for panel, ax, title in zip("ABCD", axs.flat, titles):
    ax.set_title(title, loc="left", pad=19)
    ax.text(-0.34, 1.16, panel, transform=ax.transAxes, fontsize=10.3,
            weight="bold", va="bottom", ha="left", clip_on=False)
    ax.axvline(0, color="#777777", linewidth=0.85, zorder=0)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="y", length=0, pad=6)
    ax.tick_params(axis="x", length=3)
    ax.set_axisbelow(True)
    ax.grid(axis="x", color="#eeeeee", linewidth=0.5)
    ax.set_xlabel("RBP coefficient (95% CI)", labelpad=7)

A, B, C, D = axs.flat
for model, y in [("original", 1), ("plus_log_length", 0)]:
    for cell in COLOR:
        r = mark(A, y, "occupancy", {"cell":cell,"model":model,"cluster":"gene"}, panel="A")
        assert r["n_ORFs"] == record("occupancy",cell=cell,model="original",cluster="gene")["n_ORFs"]
A.set_yticks([1,0], ["Original\ncovariates", "Plus host\nRNA length"])
A.set_ylim(-0.45,1.45)
A.set_xlim(0,0.76)
A.set_xticks([0,0.2,0.4,0.6])
an = [record("occupancy",cell=c,model="plus_log_length",cluster="gene")["n_ORFs"] for c in COLOR]
A.text(0, 1.04, f"n = {an[0]:,} / {an[1]:,} ORFs", transform=A.transAxes,
       fontsize=7.2, color="#555555", va="bottom")

for y, outcome in [(1,"pLDDT"),(0,"catGRANULE")]:
    for cell in COLOR:
        r = mark(B,y,"structural", {"cell":cell,"model":"plus_log_length",
                                   "scheme":"transcript","outcome":outcome},
                 panel="B",standardized=True,ci=90)
        assert r["n"] == record("structural",cell=cell,model="plus_log_length",scheme="transcript",outcome="pLDDT")["n"]
for x in [-BOUND, BOUND]:
    B.axvline(x, color="#888888", linestyle=(0,(3,2)), linewidth=0.9, zorder=0)
B.set_yticks([1,0], ["pLDDT", "catGRANULE"])
B.set_ylim(-0.45,1.45)
B.set_xlim(-0.125,0.125)
B.set_xticks([-0.1,0,0.1])
B.set_xticklabels(["-0.10","0","+0.10"])
B.set_xlabel("Standardized RBP coefficient\n(SD per SD; 90% CI)",labelpad=6)
bn = [record("structural",cell=c,model="plus_log_length",scheme="transcript",outcome="pLDDT")["n"] for c in COLOR]
assert bn[0] == bn[1]
B.text(0,1.04,f"n = {bn[0]:,} per cell; dashed: ±{BOUND:.2f}",transform=B.transAxes,
       fontsize=7.1,color="#555555",va="bottom")

base={"stratum":"uORF_primary","population":"corrected_reselected",
      "reference_version":"corrected","length_definition":"corrected_CDS_codon",
      "adjustment":"plus_log_length","cluster":"enst"}
for y, spec in [(2,"nc_rpf_unadjusted_for_cds"),(1,"canonical_cds_rpf"),(0,"allocation_cds_adjusted")]:
    for cell in COLOR:
        r=mark(C,y,"within_host",dict(base,cell=cell,specification=spec),panel="C")
        assert r["n"] == record("within_host",**dict(base,cell=cell,specification="allocation_cds_adjusted"))["n"]
C.set_yticks([2,1,0],["uORF","Canonical CDS","uORF, CDS-\nadjusted"])
C.set_ylim(-0.45,2.45)
C.set_xlim(-0.30,1.05)
C.set_xticks([-0.2,0,0.4,0.8])
cn = [record("within_host",**dict(base,cell=c,specification="allocation_cds_adjusted"))["n"] for c in COLOR]
C.text(0,1.04,f"n = {cn[0]:,} / {cn[1]:,} paired uORFs",transform=C.transAxes,
       fontsize=7.2,color="#555555",va="bottom")

for cell in COLOR:
    r=mark(D,0,"within_host",dict(base,cell=cell,specification="density_ratio"),panel="D")
    assert r["n"] == record("within_host",**dict(base,cell=cell,specification="allocation_cds_adjusted"))["n"]
D.set_yticks([0],["uORF / CDS"])
D.set_ylim(-0.65,0.65)
D.set_xlim(-0.72,0.08)
D.set_xticks([-0.6,-0.4,-0.2,0])
D.text(0,1.04,"Same paired subset as panel C",transform=D.transAxes,
       fontsize=7.2,color="#555555",va="bottom")

legend=[Line2D([],[],marker="o" if cell=="K562" else "s",linestyle="none",
               markersize=4.5,color=color,label=cell) for cell,color in COLOR.items()]
fig.legend(handles=legend,loc="upper center",bbox_to_anchor=(0.55,0.985),
           frameon=False,ncol=2,fontsize=8,handletextpad=0.5,columnspacing=2.5)

# Check every nonempty visible text bbox before export; never use tight bbox to hide overflow.
fig.canvas.draw()
renderer=fig.canvas.get_renderer()
figbox=fig.bbox
overflow=[]
for obj in fig.findobj(match=Text):
    if obj.get_visible() and obj.get_text().strip():
        b=obj.get_window_extent(renderer=renderer)
        if b.x0<figbox.x0-0.5 or b.y0<figbox.y0-0.5 or b.x1>figbox.x1+0.5 or b.y1>figbox.y1+0.5:
            overflow.append(obj.get_text())
if overflow:
    raise ValueError({"text_outside_canvas":overflow})
fig.savefig(OUT/"Figure_5.png",dpi=300,facecolor="white")
fig.savefig(OUT/"Figure_5.pdf",facecolor="white",metadata={"Title":"Host-RNA length and same-transcript sensitivity analyses","Author":"Kyota Yasuda","Subject":"Figure 5; verified results, no new fits"})
plt.close(fig)

from PIL import Image
with Image.open(OUT/"Figure_5.png") as im:
    width,height=im.size
    dpi=im.info.get("dpi")
    if width>2044 or width<2042:
        raise ValueError((width,"unexpected publication width"))

claims={"purpose":"Figure 5 reads saved validated model results without refitting",
        "source_sha256":{key:sha(path) for key,path in SOURCES.items()},
        "source_paths":{key:os.path.relpath(path, ROOT) for key,path in SOURCES.items()},
        "reference_bounds":{"value":BOUND,"source":os.path.relpath(SPEC, ROOT),"sha256":sha(SPEC),"role":"prespecified conventional primary TOST margins; guides alone do not imply corrected equivalence"},
        "figure_design_constants":{"width_mm":WIDTH_MM,"height_mm":HEIGHT_MM,"dpi":300,
             "min_font_pt":7.1,"color_hex":COLOR},
        "python":sys.version,"matplotlib":matplotlib.__version__,
        "plotted_records":PLOTTED,"assertions":{"finite_and_CI_contains_coefficient":True,"text_inside_canvas":True,"plotted_records":len(PLOTTED)},
        "outputs":{"Figure_5.png":{"sha256":sha(OUT/"Figure_5.png"),"size_pixels":[width,height],"dpi":dpi},
                   "Figure_5.pdf":{"sha256":sha(OUT/"Figure_5.pdf")}}}
(OUT/"sourceclaims.json").write_text(json.dumps(claims,ensure_ascii=False,indent=2)+"\n")
print(json.dumps({"output":str(OUT),"points":len(PLOTTED),"width_pixels":width,"height_pixels":height,"source_records_only":True},indent=2))
