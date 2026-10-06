"""
Standalone metric evaluator for SAME retargeting results.

Reads the OUT bvh produced by src/same/test.py (out/bvh/<Src>__TO__<Tgt>.bvh).
No target-GT metrics: a pair's "target" is not a real ground truth (a different
animal never performed the source motion), so mpjpe/rot_err against it are
meaningless and have been removed, along with --gt_dir. Retarget accuracy is
instead measured by model-based reconstruction / cycle, computed by
src/eval_recon_cycle.py and merged in here from recon_cycle.csv.

Metrics - OUT only (from BVH):
  jerk            [cm/s^3] Mean joint jitter (3rd-order finite difference)
  foot_skating    [cm]     Joint velocity weighted by soft contact probability
  ground_pen      [cm]     Mean ground penetration depth (0 = no penetration)

Metrics - OUT vs SOURCE (from BVH; source is the real input, not a fake GT):
  freq_alignment      [%]  PSD cosine similarity vs the source motion, x100
                           (Motion2Motion 'freq. align'; higher better).
  contact_consistency [%]  How well OUT preserves the SOURCE ground-contact timing
                           (Motion2Motion 'contact con.'; higher better).
  root_traj_err       [bl] Root-path deviation vs the source, each shifted to the
                           origin and divided by its own bbox diagonal (body-length
                           units), so different species compare fairly. Lower better.

Metrics - model-based, merged from recon_cycle.csv (src/eval_recon_cycle.py):
  recon_mpjpe/recon_rot   [cm]/[deg]  reconstruction A->A error (per unique source)
  cycle_mpjpe/cycle_rot   [cm]/[deg]  cycle A->B->A' error (per pair)
  Both have a true GT (the original source), so they are valid error metrics.

Usage:
  conda activate same
  cd /home/inseo/Github/SAME_original

  # 1) produce OUT bvh:            python src/same/test.py ...
  # 2) recon/cycle (model-based):  python src/eval_recon_cycle.py --out_csv <result_dir>/recon_cycle.csv ...
  # 3) score (merges recon_cycle.csv):
python metric/metric.py \
    --result_dir result/260803_cfg_VT_fold0/test \
    --pairs_txt  data/Trueboness_processed_byVT/processed/truebones_vt_groups_fold0_test.txt \
    --src_dir    data/Trueboness_processed_byVT/augmented
"""

import argparse
import csv
import glob
import os
import re
import sys

import numpy as np

# ---- make the local fairmotion importable (falls back to the installed one) --
_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.join(_HERE, "..", "src")
for _p in (_SRC, os.path.join(_SRC, "fairmotion")):
    _p = os.path.abspath(_p)
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)

from fairmotion.data import bvh  # noqa: E402

# ----------------------------- constants -----------------------------

FPS: float = 30.0
CONTACT_H_CM: float = 5.0   # soft contact height threshold [cm]
FREQ_MIN_HZ: float = 1.0    # freq_alignment low cutoff [Hz]; drops the
                            # fundamental, which is inseparable from drift here


# ----------------------------- BVH loading ---------------------------

def load_motion(bvh_path: str):
    """
    Load a BVH file (units: cm, produced by src/same/test.py).

    Returns
    -------
    pos  : np.ndarray [T, J, 3]      global joint positions (cm)
    rot  : np.ndarray [T, J, 3, 3]   local rotation matrices
                                      (root: global rot; others: local-to-parent)
    skel : fairmotion Skeleton
    """
    motion = bvh.load(bvh_path)
    T = len(motion.poses)
    J = motion.skel.num_joints()

    pos = np.zeros((T, J, 3), dtype=np.float32)
    rot = np.zeros((T, J, 3, 3), dtype=np.float32)

    for t, pose in enumerate(motion.poses):
        for j in range(J):
            pos[t, j] = pose.get_transform(j, local=False)[:3, 3]
            rot[t, j] = np.asarray(pose.data[j])[:3, :3]

    return pos, rot, motion.skel


# Target-GT metrics (mpjpe / root_rel_mpjpe / rot_err) were removed: the pair's
# target is not a real ground truth, so error against it is meaningless. Retarget
# accuracy is measured by reconstruction / cycle (eval_recon_cycle.py), merged in
# from recon_cycle.csv.


def _contact_signal(pos: np.ndarray) -> np.ndarray:
    """Per-frame 'groundedness' in [0,1], scale/offset/skeleton-invariant.

    Takes the lowest joint's height each frame (proxy for how close the body is to
    the ground) and min-max normalizes it over the clip, inverted so the
    most-grounded frame -> 1 and the most-airborne frame -> 0. Comparing this
    across two motions asks WHEN each is at its lowest, independent of absolute
    height, units, or body scale (which differ between source and target).
    """
    min_h = pos[:, :, 1].min(axis=1)             # [T] lowest joint height per frame
    g = -min_h                                    # higher = more grounded
    lo, hi = float(g.min()), float(g.max())
    return (g - lo) / (hi - lo + 1e-8)            # [T] in [0,1]


def compute_contact_consistency(out_pos: np.ndarray, src_pos: np.ndarray = None) -> float:
    """Contact consistency [%]: how well the OUTPUT preserves the SOURCE's
    ground-contact timing (Motion2Motion 'contact con.').

    Compares the per-frame groundedness signal (see _contact_signal) of source and
    output and reports 100*(1 - mean|c_src - c_out|). Higher is better: 100 =
    identical grounding pattern over time. Because each signal is min-max
    normalized per motion, this is invariant to absolute height, units, and body
    scale, so the meter-unit dataset source bvh compares fine against the cm
    output. Height-based proxy (no manual contact-bone labels). Needs the source;
    returns nan if unavailable.
    """
    if src_pos is None or len(out_pos) < 2 or len(src_pos) < 2:
        return float("nan")
    T = min(len(out_pos), len(src_pos))
    c_out = _contact_signal(out_pos[:T])
    c_src = _contact_signal(src_pos[:T])
    return float((1.0 - np.abs(c_src - c_out).mean()) * 100.0)


def _body_scale(pos: np.ndarray) -> float:
    """Character size = bbox diagonal of the (root-relative) rest skeleton.

    Uses frame 0, joints relative to the root, so it measures body size (not the
    travelled path) and is invariant to where the character starts. Different
    species have very different sizes, so dividing the root path by this puts both
    source and output in dimensionless body-length units.
    """
    p0 = pos[0] - pos[0, 0:1]                     # frame 0, root-relative [J,3]
    ext = p0.max(axis=0) - p0.min(axis=0)         # bbox side lengths
    return float(np.linalg.norm(ext) + 1e-8)      # bbox diagonal


def compute_root_traj_err(out_pos: np.ndarray, src_pos: np.ndarray = None) -> float:
    """Root trajectory error [body-lengths]: how far the OUTPUT root path deviates
    from the SOURCE root path, size-normalized so different species are comparable.

    Each root trajectory is shifted to start at the origin (frame-0 root removed)
    and divided by that character's own bbox diagonal (see _body_scale), giving a
    dimensionless path in body-lengths. Error = mean over frames of the 3D
    distance between the two normalized root paths. Lower is better (0 = same
    relative path). Needs the source; nan if unavailable.
    """
    if src_pos is None or len(out_pos) < 2 or len(src_pos) < 2:
        return float("nan")
    T = min(len(out_pos), len(src_pos))
    ts = (src_pos[:T, 0] - src_pos[0, 0]) / _body_scale(src_pos)   # [T,3] body-lengths
    to = (out_pos[:T, 0] - out_pos[0, 0]) / _body_scale(out_pos)
    return float(np.linalg.norm(ts - to, axis=-1).mean())


# ------------------------------ No-GT metrics ------------------------

def _psd_over_freq(pos: np.ndarray, root_relative: bool = False) -> np.ndarray:
    """Power Spectral Density aggregated over all joints & axes.

    pos : [T, J, 3] global joint positions.
    With root_relative the root joint is subtracted first, so the spectrum
    describes articulation rather than the global path. The per-joint/axis mean
    (DC) is removed, an rFFT is taken along time, and the power |rfft|^2 is
    summed over joints and axes -> a vector indexed by frequency bin (length
    T//2+1). Its length depends only on T, not on the joint count.
    """
    if root_relative:
        pos = pos - pos[:, :1]                      # drop the global path
    x = pos - pos.mean(axis=0, keepdims=True)       # remove DC per joint/axis
    F = np.fft.rfft(x, axis=0)                        # [nfreq, J, 3] complex
    # breakpoint()
    return (np.abs(F) ** 2).sum(axis=(1, 2))         # [nfreq]


def compute_jerk(pos: np.ndarray, fps: float = FPS) -> float:
    """
    Mean joint position jitter via 3rd-order finite difference [cm/s^3].
    Ref: https://en.wikipedia.org/wiki/Finite_difference_coefficient
    """
    if len(pos) < 4:
        return float("nan")
    j3 = (pos[3:] - 3 * pos[2:-1] + 3 * pos[1:-2] - pos[:-3]) * (fps ** 3)
    return float(np.linalg.norm(j3, axis=-1).mean())


def compute_foot_skating(pos: np.ndarray, H: float = CONTACT_H_CM) -> float:
    """
    Foot sliding metric [cm].
    All joints contribute weighted by soft contact probability:
      contact(h) = clamp(2 - 2^(h/H), 0, 1)
    Ref: Mode-Adaptive Neural Networks for Quadruped Motion Control.
    """
    if len(pos) < 2:
        return float("nan")
    vel = np.linalg.norm(pos[1:] - pos[:-1], axis=-1)      # [T-1, J]
    h = pos[1:, :, 1]                                        # y-height [T-1, J]
    contact = np.clip(2.0 - np.power(2.0, h / H), 0.0, 1.0)
    return float((vel * contact).mean())


def compute_ground_pen(pos: np.ndarray) -> float:
    """
    Ground penetration depth [cm].
    Mean depth of joints below y=0 (clipped to >=0; higher = worse).
    """
    pen = np.maximum(-pos[..., 1], 0.0)
    return float(pen.mean())


def compute_freq_alignment(
    out_pos: np.ndarray,
    src_pos: np.ndarray = None,
    fps: float = None,
    f_min: float = None,
    root_relative: bool = True,
) -> float:
    """Frequency alignment [%], after Motion2Motion (Table 1, 'freq. align').

    Temporal alignment between the SOURCE and the retargeted OUTPUT, measured as
    the cosine similarity of their PSD (aggregated across all joints), x100.
    Higher is better: 100 = identical frequency content / cadence.

    Two restrictions decide which part of the spectrum is compared:
      root_relative : subtract the root joint, so the global path drops out
      f_min         : keep only bins at or above this frequency [Hz]
    Motion2Motion states the metric as a plain PSD cosine similarity over global
    positions (Sec. 4.2). Measured that way it has almost no dynamic range on
    this data: 80% of a source clip's power sits below 1 Hz, and an unrelated
    motion scores as high as a correct retarget. Be precise about what the
    cutoff buys, though. These clips hold only 1-2 gait cycles (measured stride
    rate 0.38-1.0 Hz, median 0.75), so the fundamental shares bins 1-2 with
    whole-clip drift and no band choice can separate the two. f_min=1 therefore
    drops the fundamental and compares the harmonics -- the shape of the action,
    not its cadence. That separates a correct retarget from an unrelated one
    better than the unrestricted form, but cadence agreement itself is not
    spectrally measurable at this clip length; use contact-event timing for
    that. Pass root_relative=False, f_min=0 for the paper's form.

    Cosine similarity ignores magnitude and the PSD is summed over joints, so
    this is invariant to joint count and to unit scale (a meter-unit dataset
    source BVH compares fine against a cm output). It stays blind to phase: a
    time-reversed copy scores 100, since |rfft| is unchanged by time reversal.
    Requires the source motion; returns nan if none is available.
    """
    fps = FPS if fps is None else fps
    f_min = FREQ_MIN_HZ if f_min is None else f_min
    if src_pos is None or len(out_pos) < 4 or len(src_pos) < 4:
        return float("nan")
    T = min(len(out_pos), len(src_pos))              # align lengths -> same bins
    a = _psd_over_freq(out_pos[:T], root_relative)
    b = _psd_over_freq(src_pos[:T], root_relative)
    n = min(len(a), len(b))
    a, b = a[:n], b[:n]
    if f_min > 0:
        keep = np.fft.rfftfreq(T, d=1.0 / fps)[:n] >= f_min
        if not keep.any():                            # clip too short to resolve
            return float("nan")
        a, b = a[keep], b[keep]
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    # breakpoint()
    if denom == 0.0:
        return float("nan")
    return float(np.dot(a, b) / denom * 100.0)


# ------------------------- Per-pair evaluation -----------------------

def evaluate_pair(out_bvh: str, src_bvh: str = None) -> dict:
    """
    Compute the BVH-only metrics for one retarget pair.

    No target-GT metrics: the pair's "target" is not a true ground truth (a
    different animal never performed the source motion), so mpjpe/rot_err against
    it are meaningless. Retarget accuracy is instead measured by the model-based
    reconstruction / cycle metrics (see eval_recon_cycle.py), merged in from
    recon_cycle.csv at report time.

    Parameters
    ----------
    out_bvh  : path to model output BVH  (required)
    src_bvh  : path to the source BVH    (optional; freq_alignment / contact_consistency need it)

    Returns
    -------
    dict  metric_name -> float
    """
    out_pos, out_rot, _ = load_motion(out_bvh)

    src_pos = None
    if src_bvh and os.path.exists(src_bvh):
        src_pos, _, _ = load_motion(src_bvh)

    metrics: dict = {}

    # out-only metrics (always computed)
    metrics["jerk"]           = compute_jerk(out_pos)
    metrics["foot_skating"]   = compute_foot_skating(out_pos)
    metrics["ground_pen"]     = compute_ground_pen(out_pos)

    # source-vs-output metrics (nan if no source)
    metrics["freq_alignment"] = compute_freq_alignment(out_pos, src_pos)
    metrics["contact_consistency"] = compute_contact_consistency(out_pos, src_pos)
    metrics["root_traj_err"] = compute_root_traj_err(out_pos, src_pos)

    return metrics


# ------------------------- Pair discovery ----------------------------

def _resolve(path: str, result_dir: str):
    """Resolve a BVH path recorded in retarget_log.csv.

    test.py logs absolute paths, so a log generated on another machine (or after
    the result dir was moved) points nowhere. Fall back to the same basename
    inside result_dir. Returns None if nothing resolves.
    """
    if not path:
        return None
    if os.path.exists(path):
        return path
    local = os.path.join(result_dir, os.path.basename(path))
    return local if os.path.exists(local) else None


def _resolve_src(out_bvh, src_rel, result_dir, src_dir=None):
    """Find the SOURCE BVH for one pair (needed by freq_alignment / contact_consistency):
      1. a <stem>__SRC.bvh next to OUT / in result_dir (older test.py runs)
      2. --src_dir / <src_rel with .npz->.bvh>   (dataset bvh; units don't matter,
         both source metrics are scale-invariant)
    The source is the real input motion (not a fake target GT), so locating it is
    legitimate. Returns a path that exists, or None.
    """
    stem_out = os.path.basename(out_bvh)
    if stem_out.endswith("__OUT.bvh"):
        stem = stem_out[: -len("__OUT.bvh")]
        for cand in (os.path.join(os.path.dirname(out_bvh), stem + "__SRC.bvh"),
                     os.path.join(result_dir, stem + "__SRC.bvh")):
            if os.path.exists(cand):
                return cand
    if src_dir and src_rel and src_rel != "?":
        rel_bvh = os.path.splitext(src_rel)[0] + ".bvh"
        for cand in (os.path.join(src_dir, rel_bvh),
                     os.path.join(src_dir, os.path.basename(rel_bvh))):
            if os.path.exists(cand):
                return cand
    return None


def _safe_name(rel_path: str) -> str:
    """Same stem test.py uses to name the BVH files (species__motion)."""
    base = rel_path.replace("\\", "/")
    base = "__".join(base.split("/")[-2:])
    base = re.sub(r"\.npz$", "", base)
    return re.sub(r"[^A-Za-z0-9_\-\.]+", "_", base)


def read_pairs_file(pairs_txt: str):
    """Read a SAME pair list ('src_rel.npz <tab> tgt_rel.npz' per line)."""
    pairs = []
    with open(pairs_txt, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            pairs.append((parts[0], parts[1]))
    return pairs


def _find_out_bvh(result_dir: str, stem: str):
    """Locate the OUT bvh for a pair stem '<src>__TO__<tgt>'.

    New layout (test.py / collect_outputs.py): out/bvh/<stem>.bvh
    Old layout: (any subdir)/pair<idx>__<stem>__OUT.bvh
    Returns a path that exists, or None.
    """
    cand = os.path.join(result_dir, "out", "bvh", f"{stem}.bvh")
    if os.path.exists(cand):
        return cand
    hits = sorted(glob.glob(os.path.join(result_dir, "**", f"*{stem}__OUT.bvh"),
                            recursive=True))
    return hits[0] if hits else None


def find_pairs_from_txt(result_dir: str, pairs_txt: str, src_dir: str = None):
    """
    Drive evaluation from a pair list (e.g. truebones_test.txt). For each 'src tgt'
    line the matching OUT bvh is located by its 'src__TO__tgt' stem (new layout
    out/bvh/<stem>.bvh, or old pair<idx>__<stem>__OUT.bvh); the source bvh is
    resolved from src_dir / beside the OUT. Only pairs whose OUT bvh exists scored.

    Returns list of (out_bvh, src_bvh_or_None, src_rel, tgt_rel).
    """
    pairs = read_pairs_file(pairs_txt)
    out = []
    n_missing_out = 0
    for src_rel, tgt_rel in pairs:
        stem = f"{_safe_name(src_rel)}__TO__{_safe_name(tgt_rel)}"
        out_bvh = _find_out_bvh(result_dir, stem)
        if out_bvh is None:
            n_missing_out += 1
            continue
        src_bvh = _resolve_src(out_bvh, src_rel, result_dir, src_dir)
        out.append((out_bvh, src_bvh, src_rel, tgt_rel))
    if n_missing_out:
        print(f"[pairs_txt] {n_missing_out}/{len(pairs)} pairs had no OUT.bvh "
              f"in {result_dir} (skipped)")
    return out


def find_pairs(result_dir: str, src_dir: str = None):
    """
    Returns list of (out_bvh, src_bvh_or_None, src_rel, tgt_rel).
    Prefers retarget_log.csv produced by src/same/test.py;
    falls back to globbing OUT bvh.
    """
    log_path = os.path.join(result_dir, "retarget_log.csv")

    if os.path.exists(log_path):
        pairs = []
        with open(log_path, newline="") as f:
            for row in csv.DictReader(f):
                if row.get("status") != "OK":
                    continue
                src_rel = row.get("src_rel", "?")
                tgt_rel = row.get("tgt_rel", "?")
                # logged path first; else the new out/bvh/<stem>.bvh layout
                out_bvh = _resolve(row.get("out_bvh", ""), result_dir)
                if out_bvh is None:
                    stem = f"{_safe_name(src_rel)}__TO__{_safe_name(tgt_rel)}"
                    out_bvh = _find_out_bvh(result_dir, stem)
                if out_bvh is None:  # nothing to score for this pair
                    continue
                src_bvh = _resolve_src(out_bvh, src_rel, result_dir, src_dir)
                pairs.append((out_bvh, src_bvh, src_rel, tgt_rel))
        return pairs

    # Fallback: no retarget_log.csv -> glob OUT bvh (new layout, then old)
    out_files = sorted(glob.glob(os.path.join(result_dir, "out", "bvh", "*.bvh")))
    if not out_files:
        out_files = sorted(glob.glob(
            os.path.join(result_dir, "**", "*__OUT.bvh"), recursive=True))
    pairs = []
    for out_bvh in out_files:
        base = os.path.basename(out_bvh)
        stem = re.sub(r"^pair\d+__", "", base)
        stem = stem[: -len("__OUT.bvh")] if stem.endswith("__OUT.bvh") else stem[: -len(".bvh")]
        src_bvh = _resolve_src(out_bvh, None, result_dir, src_dir)
        pairs.append((out_bvh, src_bvh, stem, ""))
    return pairs


# ------------------------------- main --------------------------------

OUT_KEYS   = ["jerk", "foot_skating", "ground_pen"]     # out-only, from BVH
SRC_KEYS   = ["freq_alignment", "contact_consistency", "root_traj_err"]  # out-vs-source, from BVH
MODEL_KEYS = ["recon_mpjpe", "recon_rot", "cycle_mpjpe", "cycle_rot"]  # merged from recon_cycle.csv
ALL_KEYS   = OUT_KEYS + SRC_KEYS + MODEL_KEYS
UNITS      = {
    "jerk": "cm/s^3", "foot_skating": "cm", "ground_pen": "cm",
    "freq_alignment": "%",
    "contact_consistency": "%",
    "root_traj_err": "bl",
    "recon_mpjpe": "cm", "recon_rot": "deg",
    "cycle_mpjpe": "cm", "cycle_rot": "deg",
}


FOOTLESS_SPECIES = ["Anaconda", "KingCobra"]   # limbless: foot_skating not meaningful


def _target_species(out_bvh: str, tgt_rel: str = "") -> str:
    """Species of the OUTPUT skeleton: the folder of tgt_rel, else parsed from the
    '<Src>__TO__<TgtSpecies>__<TgtAction>' file name."""
    if tgt_rel and tgt_rel != "?":
        return tgt_rel.replace("\\", "/").split("/")[0]
    stem = os.path.basename(out_bvh)
    if "__TO__" not in stem:
        return ""
    return stem.split("__TO__", 1)[1].split("__", 1)[0]


def load_recon_cycle(result_dir):
    """Read recon_cycle.csv (from eval_recon_cycle.py) if present in result_dir.

    Returns (cycle_by_pair, recon_by_src, found):
      cycle_by_pair[(src_rel, tgt_rel)] = (cycle_mpjpe, cycle_rot)
      recon_by_src[src_rel]             = (recon_mpjpe, recon_rot)
    """
    path = os.path.join(result_dir, "recon_cycle.csv")
    cycle_by_pair, recon_by_src = {}, {}
    if not os.path.exists(path):
        return cycle_by_pair, recon_by_src, False

    def _num(v):
        try:
            return float(v)
        except (TypeError, ValueError):
            return None

    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            if row.get("idx") == "MEAN":
                continue
            sr, tr = row.get("src_rel", ""), row.get("tgt_rel", "")
            cm, cr = _num(row.get("cycle_mpjpe")), _num(row.get("cycle_rot"))
            if cm is not None:
                cycle_by_pair[(sr, tr)] = (cm, cr)
            rm, rr = _num(row.get("recon_mpjpe")), _num(row.get("recon_rot"))
            if rm is not None:
                recon_by_src[sr] = (rm, rr)
    return cycle_by_pair, recon_by_src, True


def main():
    global FPS, CONTACT_H_CM, FREQ_MIN_HZ

    parser = argparse.ArgumentParser(
        description="Evaluate SAME retargeting metrics from BVH files"
    )
    parser.add_argument("--result_dir", type=str, required=True,
                        help="Result dir with out/bvh/<Src>__TO__<Tgt>.bvh "
                             "(and optionally recon_cycle.csv to merge)")
    parser.add_argument("--pairs_txt", type=str, default=None,
                        help="Pair list (e.g. data/.../processed/truebones_test.txt). "
                             "Drives which pairs to score, matching OUT.bvh by its "
                             "src__TO__tgt stem. Use instead of retarget_log.csv "
                             "(e.g. to score only the test split).")
    parser.add_argument("--src_dir", type=str, default=None,
                        help="Where to find the SOURCE BVH (real input motion, not "
                             "a target GT) for freq_alignment / contact_consistency. "
                             "Dataset bvh dir, matched by src_rel. Both metrics are "
                             "scale-invariant, so units don't matter. If omitted with "
                             "--pairs_txt, defaults to the dataset bvh next to it.")
    parser.add_argument("--footless", type=str, nargs="*", default=FOOTLESS_SPECIES,
                        help="Target species excluded from foot_skating (limbless "
                             "bodies slide by design). Pass with no names to disable.")
    parser.add_argument("--out_csv",    type=str, default=None,
                        help="Output CSV (default: <result_dir>/metrics.csv)")
    parser.add_argument("--fps",       type=float, default=FPS)
    parser.add_argument("--contact_h", type=float, default=CONTACT_H_CM,
                        help="Contact height threshold [cm] for foot_skating")
    parser.add_argument("--freq_min", type=float, default=FREQ_MIN_HZ,
                        help="Low cutoff [Hz] for freq_alignment. Bins below it "
                             "mix the action's fundamental with whole-clip "
                             "drift and are not separable at this clip length; "
                             "0 disables the cutoff (freq_alignment_raw is "
                             "always the unrestricted global-position score).")
    args = parser.parse_args()

    FPS = args.fps
    CONTACT_H_CM = args.contact_h
    FREQ_MIN_HZ = args.freq_min

    out_csv = args.out_csv or os.path.join(args.result_dir, "metrics.csv")
    os.makedirs(os.path.dirname(os.path.abspath(out_csv)), exist_ok=True)

    # locate the SOURCE bvh dir (real input, for freq_alignment / contact_consistency)
    src_dir = args.src_dir
    if src_dir is None and args.pairs_txt:
        cand = os.path.join(os.path.dirname(os.path.dirname(
            os.path.abspath(args.pairs_txt))), "bvh")
        if os.path.isdir(cand):
            src_dir = cand
            print(f"[metric] --src_dir not set; using dataset bvh: {src_dir}")

    if args.pairs_txt:
        pairs = find_pairs_from_txt(args.result_dir, args.pairs_txt, src_dir=src_dir)
    else:
        pairs = find_pairs(args.result_dir, src_dir=src_dir)

    if not pairs:
        print(f"[ERROR] No pairs found in: {args.result_dir}")
        sys.exit(1)

    # recon / cycle metrics come from eval_recon_cycle.py (model-based), merged here
    cycle_by_pair, recon_by_src, rc_found = load_recon_cycle(args.result_dir)
    n_src = sum(1 for _, src, _, _ in pairs if src)
    print(f"[metric] {len(pairs)} pairs | source resolved {n_src}/{len(pairs)} "
          f"(freq_alignment/contact_consistency)")
    if rc_found:
        print(f"[metric] recon_cycle.csv merged: {len(recon_by_src)} recon srcs, "
              f"{len(cycle_by_pair)} cycle pairs")
    else:
        print(f"[metric] recon_cycle.csv NOT found in {args.result_dir} "
              f"-> recon/cycle = N/A (run src/eval_recon_cycle.py)")
    print(f"[metric] fps={FPS} contact_h={CONTACT_H_CM}cm "
          f"freq_min={FREQ_MIN_HZ}Hz (root-relative)")
    print(f"[metric] result_dir = {args.result_dir}\n")

    rows = []
    agg = {k: [] for k in ALL_KEYS}

    def _fmt(v):
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return "N/A"
        return f"{v:.4f}"

    def _pct(mm, key):
        v = mm.get(key)
        return f"{v:.2f}%" if (v is not None and not np.isnan(v)) else "n/a"

    footless = set(args.footless)
    n_footless = 0

    for i, (out_bvh, src_bvh, src_rel, tgt_rel) in enumerate(pairs):
        label = f"{src_rel} -> {tgt_rel}" if tgt_rel else src_rel
        if not os.path.exists(out_bvh):
            print(f"  [{i:03d}] SKIP (missing): {out_bvh}")
            continue

        try:
            m = evaluate_pair(out_bvh, src_bvh)
        except Exception as exc:
            print(f"  [{i:03d}] ERROR: {label} -- {exc}")
            continue

        # foot_skating assumes planted contacts stay still; a limbless body slides
        # along the ground by design, so skip it when the OUTPUT skeleton is one.
        if _target_species(out_bvh, tgt_rel) in footless:
            m["foot_skating"] = float("nan")
            n_footless += 1

        # merge model-based recon / cycle from recon_cycle.csv
        if src_rel in recon_by_src:
            m["recon_mpjpe"], m["recon_rot"] = recon_by_src[src_rel]
        if (src_rel, tgt_rel) in cycle_by_pair:
            m["cycle_mpjpe"], m["cycle_rot"] = cycle_by_pair[(src_rel, tgt_rel)]

        row = {"idx": i, "label": label}
        row.update({k: _fmt(m.get(k)) for k in ALL_KEYS})
        rows.append(row)

        for k in ALL_KEYS:
            v = m.get(k)
            if v is not None and not (isinstance(v, float) and np.isnan(v)):
                agg[k].append(v)

        out_str = (f"jerk={m['jerk']:.2f}  fs={_fmt(m['foot_skating'])}cm  "
                   f"gp={m['ground_pen']:.4f}cm")
        src_str = (f"freq_align={_pct(m,'freq_alignment')} "
                   f"contact_con={_pct(m,'contact_consistency')} "
                   f"root_traj={_fmt(m.get('root_traj_err'))}bl")
        rc_str = (f"recon={_fmt(m.get('recon_mpjpe'))}cm/{_fmt(m.get('recon_rot'))}deg  "
                  f"cycle={_fmt(m.get('cycle_mpjpe'))}cm/{_fmt(m.get('cycle_rot'))}deg")
        print(f"  [{i:03d}] {label}")
        print(f"         out  : {out_str}")
        print(f"         src  : {src_str}")
        print(f"         model: {rc_str}")

    # recon is per unique source -> aggregate over unique sources (avoid counting a
    # source once per pair it appears in); cycle stays per-pair.
    agg["recon_mpjpe"] = [v[0] for v in recon_by_src.values() if v[0] is not None]
    agg["recon_rot"]   = [v[1] for v in recon_by_src.values() if v[1] is not None]

    if n_footless:
        print(f"\n[metric] foot_skating skipped for {n_footless} pairs with a "
              f"limbless target ({', '.join(sorted(footless))})")

    # -- write CSV --
    fieldnames = ["idx", "label"] + ALL_KEYS
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
        summary = {"idx": "MEAN", "label": f"n={len(rows)}"}
        summary.update({k: f"{np.mean(agg[k]):.4f}" if agg[k] else "N/A" for k in ALL_KEYS})
        w.writerow(summary)

    print(f"\n[metric] -> {out_csv}")
    print("\n================ Summary ================")
    for k in ALL_KEYS:
        if agg[k]:
            print(f"  {k:<20s} {np.mean(agg[k]):9.4f}  [{UNITS.get(k,'')}]  (n={len(agg[k])})")
        else:
            print(f"  {k:<20s} N/A")


if __name__ == "__main__":
    main()
