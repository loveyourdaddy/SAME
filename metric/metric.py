"""
Standalone metric evaluator for SAME retargeting results.
Ported from any2any (SAME/metric/metric.py) into this repo.

Computes metrics from paired (OUT.bvh, TGT.bvh) files in a result directory,
i.e. the BVH files produced by src/same/test.py (which writes
  <out_dir>/pair<idx>__..__SRC/TGT/OUT.bvh  +  retarget_log.csv ).

Metrics - With pseudo-GT (OUT vs TGT, same skeleton):
  mpjpe           [cm]     Global Mean Per-Joint Position Error
  root_rel_mpjpe  [cm]     Root-relative Mean Per-Joint Position Error
  rot_err         [deg]    Local joint rotation geodesic error
  contact_consistency [cm]  Mean contact consistency (0 = no contact)

Metrics - OUT only:
  jerk            [cm/s^3] Mean joint jitter (3rd-order finite difference)
  foot_skating    [cm]     Joint velocity weighted by soft contact probability
  ground_pen      [cm]     Mean ground penetration depth (0 = no penetration)

Metrics - needs SOURCE (OUT vs SRC):
  freq_alignment  [%]      PSD cosine similarity vs the source motion, x100
                           (Motion2Motion 'freq. align'; 100 = identical spectrum,
                           higher is better). Needs the source; scale/joint-count
                           invariant, so the dataset source bvh works directly.
                           Taken on root-relative positions over bins >= 1 Hz
                           (--freq_min); see compute_freq_alignment for what
                           that cutoff does and does not measure.
  freq_alignment_raw [%]   The same on global positions with no band cutoff, i.e.
                           the paper's unrestricted form. Reported for reference
                           only: it is dominated by whole-clip drift and cannot
                           tell a correct retarget from an unrelated motion.

Prerequisites:
  BVH files must be saved with consistent cm units - src/same/test.py applies
  scale_motion(..., unit_scale=100) to both skeleton offsets and root
  translation, so the default test output already satisfies this.

Usage:
  conda activate same
  cd /home/inseo/Github/SAME_original

# 1) produce BVH + retarget_log.csv
# 2) score every OK pair from that run (GT = the TGT.bvh test.py saved)
cd ..
python metric/metric.py \
    --result_dir result/260803_cfg_VT_split/test \
    --gt_dir result/260803_cfg_VT_split/test \
    --pairs_txt  data/Trueboness_processed_byVT/processed/truebones_vt_exact_test.txt \
    --out_csv result/260803_cfg_VT_split/test/metrics_test.csv
# 3) render: render파일 수정후
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


# ------------------------- GT-required metrics -----------------------

def compute_mpjpe(out_pos: np.ndarray, tgt_pos: np.ndarray) -> float:
    """Mean Per-Joint Position Error [cm]."""
    T = min(len(out_pos), len(tgt_pos))
    J = min(out_pos.shape[1], tgt_pos.shape[1])
    err = np.linalg.norm(out_pos[:T, :J] - tgt_pos[:T, :J], axis=-1)
    return float(err.mean())


def compute_root_rel_mpjpe(out_pos: np.ndarray, tgt_pos: np.ndarray) -> float:
    """Root-relative MPJPE [cm]: subtract root joint position before comparing."""
    T = min(len(out_pos), len(tgt_pos))
    J = min(out_pos.shape[1], tgt_pos.shape[1])
    out_rel = out_pos[:T, :J] - out_pos[:T, :1]
    tgt_rel = tgt_pos[:T, :J] - tgt_pos[:T, :1]
    return float(np.linalg.norm(out_rel - tgt_rel, axis=-1).mean())


def compute_rot_err(out_rot: np.ndarray, tgt_rot: np.ndarray) -> float:
    """
    Geodesic rotation error [degrees].
    angle = arccos( (trace(R_out^T @ R_tgt) - 1) / 2 )
    """
    T = min(len(out_rot), len(tgt_rot))
    J = min(out_rot.shape[1], tgt_rot.shape[1])
    R_o = out_rot[:T, :J]
    R_t = tgt_rot[:T, :J]
    R_diff = np.einsum("...ij,...ik->...jk", R_o, R_t)   # R_out^T @ R_tgt
    trace = R_diff[..., 0, 0] + R_diff[..., 1, 1] + R_diff[..., 2, 2]
    cos_a = np.clip((trace - 1.0) / 2.0, -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_a)).mean())


def compute_contact_consistency(
    out_pos: np.ndarray,
    tgt_pos: np.ndarray,
    H: float = CONTACT_H_CM,
) -> float:
    """
    Contact consistency [cm].
    Applies TGT-derived soft contact weights to OUT joint velocities.
    Measures how much the output moves during frames where TGT has contact.
    Lower is better; 0 = no movement during contact = perfect consistency.
    """
    T = min(len(out_pos), len(tgt_pos))
    if T < 2:
        return float("nan")
    h_tgt = tgt_pos[1:T, :, 1]                                   # y-height of TGT
    c_tgt = np.clip(2.0 - np.power(2.0, h_tgt / H), 0.0, 1.0)   # soft contact [T-1, J]
    v_out = np.linalg.norm(out_pos[1:T] - out_pos[:T-1], axis=-1) # OUT speed [T-1, J]
    return float((v_out * c_tgt).mean())


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

def evaluate_pair(out_bvh: str, tgt_bvh: str = None, src_bvh: str = None,
                  gt_scale: float = 1.0) -> dict:
    """
    Compute all metrics for one retarget pair.

    Parameters
    ----------
    out_bvh  : path to model output BVH  (required)
    tgt_bvh  : path to ground-truth BVH  (optional; GT metrics skipped if None)
    src_bvh  : path to the source BVH    (optional; freq_alignment needs it)
    gt_scale : multiply GT positions by this factor before comparing. Use 100
               when the GT BVH is in meters while OUT is in cm (e.g. raw dataset
               BVH). Leave at 1.0 when GT is already cm (a TGT.bvh from test.py).
               (freq_alignment is scale-invariant, so src units don't matter.)

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
    # source-vs-output frequency alignment (nan if no source)
    metrics["freq_alignment"] = compute_freq_alignment(out_pos, src_pos)
    metrics["freq_alignment_raw"] = compute_freq_alignment(
        out_pos, src_pos, f_min=0.0, root_relative=False)

    # GT metrics
    if tgt_bvh and os.path.exists(tgt_bvh):
        tgt_pos, tgt_rot, _ = load_motion(tgt_bvh)
        if gt_scale != 1.0:
            tgt_pos = tgt_pos * gt_scale     # rotation is scale-invariant
        metrics["mpjpe"]                = compute_mpjpe(out_pos, tgt_pos)
        metrics["root_rel_mpjpe"]       = compute_root_rel_mpjpe(out_pos, tgt_pos)
        metrics["rot_err"]              = compute_rot_err(out_rot, tgt_rot)
        metrics["contact_consistency"]  = compute_contact_consistency(out_pos, tgt_pos)

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


def _resolve_gt(out_bvh, logged_tgt, result_dir, tgt_rel=None, gt_dir=None):
    """Find the ground-truth BVH for one pair, trying (in order):
      1. the tgt_bvh logged in retarget_log.csv (as-is or by basename in result_dir)
      2. a <stem>__TGT.bvh sitting next to the OUT file
      3. --gt_dir / <same pair stem>__TGT.bvh   (another run that saved TGT.bvh)
      4. --gt_dir / <tgt_rel with .npz->.bvh>   (raw dataset bvh; may need gt_scale)
    Returns a path that exists, or None.
    """
    tgt = _resolve(logged_tgt, result_dir)
    if tgt:
        return tgt

    stem_out = os.path.basename(out_bvh)
    if stem_out.endswith("__OUT.bvh"):
        stem = stem_out[: -len("__OUT.bvh")]
        beside = os.path.join(os.path.dirname(out_bvh), stem + "__TGT.bvh")
        if os.path.exists(beside):
            return beside
        if gt_dir:
            cand = os.path.join(gt_dir, stem + "__TGT.bvh")
            if os.path.exists(cand):
                return cand

    if gt_dir and tgt_rel and tgt_rel != "?":
        rel_bvh = os.path.splitext(tgt_rel)[0] + ".bvh"
        for cand in (os.path.join(gt_dir, rel_bvh),
                     os.path.join(gt_dir, os.path.basename(rel_bvh))):
            if os.path.exists(cand):
                return cand
    return None


def _resolve_src(out_bvh, src_rel, result_dir, gt_dir=None):
    """Find the SOURCE BVH for one pair (needed by the freq_alignment PSD metric):
      1. a <stem>__SRC.bvh next to OUT / in result_dir (older test.py runs)
      2. --gt_dir / <src_rel with .npz->.bvh>   (dataset bvh; units don't matter,
         cosine similarity is scale-invariant)
    Returns a path that exists, or None.
    """
    stem_out = os.path.basename(out_bvh)
    if stem_out.endswith("__OUT.bvh"):
        stem = stem_out[: -len("__OUT.bvh")]
        for cand in (os.path.join(os.path.dirname(out_bvh), stem + "__SRC.bvh"),
                     os.path.join(result_dir, stem + "__SRC.bvh")):
            if os.path.exists(cand):
                return cand
    if gt_dir and src_rel and src_rel != "?":
        rel_bvh = os.path.splitext(src_rel)[0] + ".bvh"
        for cand in (os.path.join(gt_dir, rel_bvh),
                     os.path.join(gt_dir, os.path.basename(rel_bvh))):
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


def find_pairs_from_txt(result_dir: str, pairs_txt: str, gt_dir: str = None):
    """
    Drive evaluation from a pair list (e.g. truebones_test.txt) instead of retarget_log.csv. 
    For each 'src tgt' line, the matching OUT.bvh in result_dir is located by its 'src__TO__tgt' stem, 
    and the GT is taken from that line's target 
    (resolved via a TGT.bvh next to OUT / in gt_dir, or the dataset bvh in gt_dir). 
    Only pairs whose OUT.bvh exists are scored.

    Returns list of (out_bvh, tgt_bvh_or_None, src_bvh_or_None, label).
    """
    pairs = read_pairs_file(pairs_txt)
    out = []
    n_missing_out = 0
    for src_rel, tgt_rel in pairs:
        stem = f"{_safe_name(src_rel)}__TO__{_safe_name(tgt_rel)}"
        hits = sorted(glob.glob(os.path.join(result_dir, f"*{stem}__OUT.bvh")))
        if not hits:
            n_missing_out += 1
            continue
        out_bvh = hits[0]
        tgt_bvh = _resolve_gt(out_bvh, "", result_dir, tgt_rel, gt_dir)
        src_bvh = _resolve_src(out_bvh, src_rel, result_dir, gt_dir)
        out.append((out_bvh, tgt_bvh, src_bvh, f"{src_rel} -> {tgt_rel}"))
    if n_missing_out:
        print(f"[pairs_txt] {n_missing_out}/{len(pairs)} pairs had no OUT.bvh "
              f"in {result_dir} (skipped)")
    return out


def find_pairs(result_dir: str, gt_dir: str = None):
    """
    Returns list of (out_bvh, tgt_bvh_or_None, src_bvh_or_None, label).
    Prefers retarget_log.csv produced by src/same/test.py;
    falls back to globbing *__OUT.bvh.
    """
    log_path = os.path.join(result_dir, "retarget_log.csv")

    if os.path.exists(log_path):
        pairs = []
        with open(log_path, newline="") as f:
            for row in csv.DictReader(f):
                if row.get("status") != "OK":
                    continue
                out_bvh = _resolve(row.get("out_bvh", ""), result_dir)
                if out_bvh is None:  # nothing to score for this pair
                    continue
                src_rel = row.get("src_rel", "?")
                tgt_rel = row.get("tgt_rel", "?")
                tgt_bvh = _resolve_gt(out_bvh, row.get("tgt_bvh", ""),
                                      result_dir, tgt_rel, gt_dir)
                src_bvh = _resolve_src(out_bvh, src_rel, result_dir, gt_dir)
                label = f"{src_rel} -> {tgt_rel}"
                pairs.append((out_bvh, tgt_bvh, src_bvh, label))
        return pairs

    # Fallback
    out_files = sorted(glob.glob(os.path.join(result_dir, "*__OUT.bvh")))
    pairs = []
    for out_bvh in out_files:
        stem = out_bvh[: -len("__OUT.bvh")]
        tgt_bvh = _resolve_gt(out_bvh, "", result_dir, None, gt_dir)
        src_bvh = _resolve_src(out_bvh, None, result_dir, gt_dir)
        label = os.path.basename(stem)
        pairs.append((out_bvh, tgt_bvh, src_bvh, label))
    return pairs


# ------------------------------- main --------------------------------

GT_KEYS    = ["mpjpe", "root_rel_mpjpe", "rot_err", "contact_consistency"]
NO_GT_KEYS = ["jerk", "foot_skating", "ground_pen", "freq_alignment",
              "freq_alignment_raw"]
ALL_KEYS   = GT_KEYS + NO_GT_KEYS
UNITS      = {
    "mpjpe": "cm", "root_rel_mpjpe": "cm", "rot_err": "deg",
    "contact_consistency": "cm",
    "jerk": "cm/s^3", "foot_skating": "cm", "ground_pen": "cm",
    "freq_alignment": "%", "freq_alignment_raw": "%",
}


def main():
    global FPS, CONTACT_H_CM, FREQ_MIN_HZ

    parser = argparse.ArgumentParser(
        description="Evaluate SAME retargeting metrics from BVH files"
    )
    parser.add_argument("--result_dir", type=str, required=True,
                        help="Directory with *__OUT.bvh / *__TGT.bvh files")
    parser.add_argument("--pairs_txt", type=str, default=None,
                        help="Pair list (e.g. data/.../processed/truebones_test.txt). "
                             "Drives which pairs to score and the GT target per pair, "
                             "matching OUT.bvh by its src__TO__tgt stem. Use instead "
                             "of reading retarget_log.csv (e.g. to score only the "
                             "test split).")
    parser.add_argument("--gt_dir", type=str, default=None,
                        help="Where to find ground-truth (TGT) BVH when result_dir "
                             "has none: another run's test dir (matched by pair "
                             "stem, cm units) or the dataset bvh dir (matched by "
                             "tgt_rel; usually needs --gt_unit_scale 100).")
    parser.add_argument("--gt_unit_scale", type=float, default=1.0,
                        help="Multiply GT positions by this before comparing. "
                             "1.0 for a cm TGT.bvh from test.py; 100 for meter "
                             "dataset bvh.")
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

    if args.pairs_txt:
        # default GT source: the dataset bvh next to the pair list
        # (data/.../motion/processed/pairs.txt -> data/.../motion/bvh), which is exactly test.py's bvh_prefix. 
        # It is in meters, so pair with --gt_unit_scale 100. 
        # For a cm-exact GT, pass --gt_dir <run>/test instead.
        gt_dir = args.gt_dir
        if gt_dir is None:
            cand = os.path.join(os.path.dirname(os.path.dirname(
                os.path.abspath(args.pairs_txt))), "bvh") # augmented
            if os.path.isdir(cand):
                gt_dir = cand
                print(f"[pairs_txt] --gt_dir not set; defaulting to dataset bvh: {gt_dir}")
                if args.gt_unit_scale == 1.0:
                    print("[pairs_txt] NOTE: dataset bvh is in meters -> pass "
                          "--gt_unit_scale 100 (and expect a small normalization "
                          "gap vs a test.py TGT.bvh).")
        pairs = find_pairs_from_txt(args.result_dir, args.pairs_txt, gt_dir=gt_dir)
    else:
        gt_dir = args.gt_dir
        pairs = find_pairs(args.result_dir, gt_dir=gt_dir)

    if not pairs:
        print(f"[ERROR] No pairs found in: {args.result_dir}")
        sys.exit(1)

    n_gt = sum(1 for _, tgt, _, _ in pairs if tgt)
    n_src = sum(1 for _, _, src, _ in pairs if src)
    print(f"[metric] GT resolved for {n_gt}/{len(pairs)} pairs"
          + (f"  (gt_dir={gt_dir}, gt_unit_scale={args.gt_unit_scale})"
             if gt_dir else ""))
    print(f"[metric] source resolved for {n_src}/{len(pairs)} pairs "
          f"(needed for freq_alignment)")

    print(f"[metric] {len(pairs)} pairs  |  fps={FPS}  contact_h={CONTACT_H_CM}cm"
          f"  freq_min={FREQ_MIN_HZ}Hz (root-relative)")
    print(f"[metric] result_dir = {args.result_dir}\n")

    rows = []
    agg = {k: [] for k in ALL_KEYS}

    def _fmt(v):
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return "N/A"
        return f"{v:.4f}"

    for i, (out_bvh, tgt_bvh, src_bvh, label) in enumerate(pairs):
        if not os.path.exists(out_bvh):
            print(f"  [{i:03d}] SKIP (missing): {out_bvh}")
            continue

        try:
            m = evaluate_pair(out_bvh, tgt_bvh, src_bvh, gt_scale=args.gt_unit_scale)
        except Exception as exc:
            print(f"  [{i:03d}] ERROR: {label} -- {exc}")
            continue

        row = {"idx": i, "label": label}
        row.update({k: _fmt(m.get(k)) for k in ALL_KEYS})
        rows.append(row)

        for k in ALL_KEYS:
            v = m.get(k)
            if v is not None and not (isinstance(v, float) and np.isnan(v)):
                agg[k].append(v)

        has_gt = "mpjpe" in m
        gt_str = (
            f"mpjpe={m['mpjpe']:.2f}cm  rr={m['root_rel_mpjpe']:.2f}cm  "
            f"rot={m['rot_err']:.2f}deg  cc={m['contact_consistency']:.4f}cm"
            if has_gt else "(no GT)"
        )
        def _pct(key):
            v = m.get(key)
            return f"{v:.2f}%" if (v is not None and not np.isnan(v)) else "n/a"

        out_str = (
            f"jerk={m['jerk']:.2f}  fs={m['foot_skating']:.4f}cm  "
            f"gp={m['ground_pen']:.4f}cm  freq_align={_pct('freq_alignment')}"
            f"  (raw {_pct('freq_alignment_raw')})"
        )
        print(f"  [{i:03d}] {label}")
        print(f"         GT   : {gt_str}")
        print(f"         out  : {out_str}")

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
