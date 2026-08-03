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

Metrics - No-GT (OUT only):
  jerk            [cm/s^3] Mean joint jitter (3rd-order finite difference)
  foot_skating    [cm]     Joint velocity weighted by soft contact probability
  ground_pen      [cm]     Mean ground penetration depth (0 = no penetration)
  freq_alignment  [Hz]     Frequency alignment of root translation (0 = perfect)

Prerequisites:
  BVH files must be saved with consistent cm units - src/same/test.py applies
  scale_motion(..., unit_scale=100) to both skeleton offsets and root
  translation, so the default test output already satisfies this.

Usage:
  conda activate same
  cd /home/inseo/Github/SAME_original

  # 1) produce BVH + retarget_log.csv
  python src/same/test.py \
      --data_dir "TruebonesZoo_processed_byJH/motion/processed/" \
      --model_epoch "260718_truebone" \
      --pairs_txt "truebones_test.txt"

  # 2) score every OK pair from that run
  python metric/metric.py --result_dir result/260718_truebone/test
  python metric/metric.py \
      --result_dir result/260718_truebone/test \
      --out_csv    result/260718_truebone/test/metrics.csv
"""

import argparse
import csv
import glob
import os
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

def _dominant_freq(signal: np.ndarray, fps: float) -> float:
    """Return the dominant frequency [Hz] of a 1-D signal via FFT."""
    n = len(signal)
    if n < 4:
        return 0.0
    sig = signal - signal.mean()
    fft_mag = np.abs(np.fft.rfft(sig))
    freqs = np.fft.rfftfreq(n, d=1.0 / fps)
    fft_mag[0] = 0.0  # suppress DC
    return float(freqs[np.argmax(fft_mag)])


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


def compute_freq_alignment(pos: np.ndarray, fps: float = FPS) -> float:
    """
    Root XZ frequency alignment [Hz].
    |dominant_freq_X - dominant_freq_Z| of root displacement.
    0 = X and Z axes share the same locomotion cadence (natural).
    Higher = axes have different dominant frequencies (unnatural retargeting).
    """
    if len(pos) < 8:
        return float("nan")
    dx = np.diff(pos[:, 0, 0])   # root X displacement per frame
    dz = np.diff(pos[:, 0, 2])   # root Z displacement per frame
    f_x = _dominant_freq(dx, fps)
    f_z = _dominant_freq(dz, fps)
    return abs(f_x - f_z)


# ------------------------- Per-pair evaluation -----------------------

def evaluate_pair(out_bvh: str, tgt_bvh: str = None) -> dict:
    """
    Compute all metrics for one retarget pair.

    Parameters
    ----------
    out_bvh : path to model output BVH  (required)
    tgt_bvh : path to ground-truth BVH  (optional; GT metrics skipped if None)

    Returns
    -------
    dict  metric_name -> float
    """
    out_pos, out_rot, _ = load_motion(out_bvh)

    metrics: dict = {}

    # No-GT metrics (always computed)
    metrics["jerk"]           = compute_jerk(out_pos)
    metrics["foot_skating"]   = compute_foot_skating(out_pos)
    metrics["ground_pen"]     = compute_ground_pen(out_pos)
    metrics["freq_alignment"] = compute_freq_alignment(out_pos)

    # GT metrics
    if tgt_bvh and os.path.exists(tgt_bvh):
        tgt_pos, tgt_rot, _ = load_motion(tgt_bvh)
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


def find_pairs(result_dir: str):
    """
    Returns list of (out_bvh, tgt_bvh_or_None, label).
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
                tgt_bvh = _resolve(row.get("tgt_bvh", ""), result_dir)
                label = f"{row.get('src_rel','?')} -> {row.get('tgt_rel','?')}"
                if out_bvh is None:  # nothing to score for this pair
                    continue
                pairs.append((out_bvh, tgt_bvh, label))
        return pairs

    # Fallback
    out_files = sorted(glob.glob(os.path.join(result_dir, "*__OUT.bvh")))
    pairs = []
    for out_bvh in out_files:
        stem = out_bvh[: -len("__OUT.bvh")]
        tgt_bvh = stem + "__TGT.bvh"
        label = os.path.basename(stem)
        pairs.append((out_bvh, tgt_bvh if os.path.exists(tgt_bvh) else None, label))
    return pairs


# ------------------------------- main --------------------------------

GT_KEYS    = ["mpjpe", "root_rel_mpjpe", "rot_err", "contact_consistency"]
NO_GT_KEYS = ["jerk", "foot_skating", "ground_pen", "freq_alignment"]
ALL_KEYS   = GT_KEYS + NO_GT_KEYS
UNITS      = {
    "mpjpe": "cm", "root_rel_mpjpe": "cm", "rot_err": "deg",
    "contact_consistency": "cm",
    "jerk": "cm/s^3", "foot_skating": "cm", "ground_pen": "cm",
    "freq_alignment": "Hz",
}


def main():
    global FPS, CONTACT_H_CM

    parser = argparse.ArgumentParser(
        description="Evaluate SAME retargeting metrics from BVH files"
    )
    parser.add_argument("--result_dir", type=str, required=True,
                        help="Directory with *__OUT.bvh / *__TGT.bvh files")
    parser.add_argument("--out_csv",    type=str, default=None,
                        help="Output CSV (default: <result_dir>/metrics.csv)")
    parser.add_argument("--fps",       type=float, default=FPS)
    parser.add_argument("--contact_h", type=float, default=CONTACT_H_CM,
                        help="Contact height threshold [cm] for foot_skating")
    args = parser.parse_args()

    FPS = args.fps
    CONTACT_H_CM = args.contact_h

    out_csv = args.out_csv or os.path.join(args.result_dir, "metrics.csv")
    os.makedirs(os.path.dirname(os.path.abspath(out_csv)), exist_ok=True)

    pairs = find_pairs(args.result_dir)
    if not pairs:
        print(f"[ERROR] No pairs found in: {args.result_dir}")
        sys.exit(1)

    print(f"[metric] {len(pairs)} pairs  |  fps={FPS}  contact_h={CONTACT_H_CM}cm")
    print(f"[metric] result_dir = {args.result_dir}\n")

    rows = []
    agg = {k: [] for k in ALL_KEYS}

    for i, (out_bvh, tgt_bvh, label) in enumerate(pairs):
        if not os.path.exists(out_bvh):
            print(f"  [{i:03d}] SKIP (missing): {out_bvh}")
            continue

        try:
            m = evaluate_pair(out_bvh, tgt_bvh)
        except Exception as exc:
            print(f"  [{i:03d}] ERROR: {label} -- {exc}")
            continue

        row = {"idx": i, "label": label}
        row.update({k: f"{m[k]:.4f}" if k in m else "N/A" for k in ALL_KEYS})
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
        no_gt_str = (
            f"jerk={m['jerk']:.2f}  fs={m['foot_skating']:.4f}cm  "
            f"gp={m['ground_pen']:.4f}cm  fa={m['freq_alignment']:.4f}Hz"
        )
        print(f"  [{i:03d}] {label}")
        print(f"         GT   : {gt_str}")
        print(f"         No-GT: {no_gt_str}")

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
