"""
cd src
python same/test.py --data_dir "train/motion/processed/" --model_epoch "250930_BIPEDS"
python same/test.py --data_dir "Trueboness_processed_byVT/processed/" --model_epoch "260803_cfg_VT_split" --pairs_txt "truebones_vt_exact_test.txt"

또는 test용 pair 파일이 따로 있다면 (pairs_txt의 모든 pair를 순회하며 retarget 후
<out_dir>/pair<idx>__..__SRC/TGT/OUT.bvh + retarget_log.csv로 저장):
"""
import argparse
import csv
import os
import re
import sys
import time
from pathlib import Path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from utils import tensor_utils
import numpy as np
import torch
from mypath import *
from same.mymodel import make_load_model
from same.mydataset import PairedDataset, get_mi_src_tgt_all_graph, npz_2_data, convert_ms_dict_r5_to_r4
from same.skel_pose_graph import SkelPoseGraph, rnd_mask
from utils.skel_gen_utils import create_random_skel
from conversions.graph_to_motion import graph_2_skel
from fairmotion.core import motion as motion_class
from fairmotion.data import bvh
from fairmotion.ops import math, conversions

# TruebonesZoo 데이터에는 __TPOSE.npz가 없어 원본 조인트 검증이 전 클립을 버림.
# 검증을 무력화해 전체 데이터를 추론에 사용 (any2any fork의 validate_joints=False와 동일).
PairedDataset.validate_joint_compatibility = lambda self, lo, filepath: True

def prepare_model_test(model_epoch, device):
    # device, printoptions
    tensor_utils.set_device(device)
    np.set_printoptions(precision=5, suppress=True)
    torch.set_printoptions(precision=5, sci_mode=False)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Model
    model, cfg = make_load_model(model_epoch, device)
    model.eval()

    load_dir = os.path.join(RESULT_DIR, model_epoch.split("/")[0])
    ms_dict = torch.load(os.path.join(load_dir, "ms_dict.pt"))
    ms_dict = convert_ms_dict_r5_to_r4(ms_dict)  # 5-dim (cos,sin,..) r -> 4-dim

    # set SkelPoseGraph class variables
    SkelPoseGraph.skel_cfg = cfg["representation"]["skel"]
    SkelPoseGraph.pose_cfg = cfg["representation"]["pose"]
    SkelPoseGraph.ms_dict = ms_dict

    return model, cfg, ms_dict


""" ================= basic functions commonly needed for tasks ================= """
from conversions.graph_to_motion import gt_recon_motion, hatD_recon_motion
from conversions.motion_to_graph import bvh_2_graph, skel_2_graph
from torch_geometric.data import Batch


def retarget(model, src_batch, tgt_batch, ms_dict, out_rep_cfg, consq_n):
    # src ground truth
    src_motion_list, src_contact_list = gt_recon_motion(src_batch, consq_n)
    
    # predicted result
    with torch.no_grad():
        z, hatD = model(src_batch, tgt_batch) # latent, decoded pose
    out_motion_list, out_contact_list = hatD_recon_motion(
        hatD, tgt_batch, out_rep_cfg, ms_dict, consq_n
    )

    # when tgt ground-truth motion is available
    if hasattr(tgt_batch, "q"):
        # tgt ground truth 있는 경우
        print("tgt ground truth available")
        tgt_motion_list, tgt_contact_list = gt_recon_motion(tgt_batch, consq_n)
        return src_motion_list[0], tgt_motion_list[0], out_motion_list[0]
    else:
        # tgt ground truth 없는 경우 (target skeleton만 있는 경우)
        print("tgt ground truth NOT available")
        tgt_skel = graph_2_skel(tgt_batch, 1)[0]
        tgt_motion = motion_class.Motion(skel=tgt_skel)
        tpose = np.eye(4)[None, ...].repeat(tgt_skel.num_joints(), 0)
        tpose[0, 1, 3] = tgt_batch.go[0, 1]  # root height
        tgt_motion.add_one_frame(tpose)

        return src_motion_list[0], tgt_motion, out_motion_list[0]


##### motion to z #####
def bvh_2_graph_z(model, bvh_filepath):
    graph_batch = bvh_2_graph(bvh_filepath).to(device=model.device)
    z = model.encoder[0](graph_batch)
    motion_list, contact_list = gt_recon_motion(graph_batch, len(z))
    return motion_list[0], graph_batch, z


##### z to motion #####
def decode_z_skel(model, z, skel, ms_dict):
    return decode_z_skelgraph(model, z, skel_2_graph(skel), ms_dict)


def decode_z_skelgraph(model, z, skel_graph, ms_dict):
    B_skel_graph = Batch.from_data_list([skel_graph] * len(z)).to(device=model.device)
    hatD = model.decoder[0](z, B_skel_graph)
    out_motion_list, out_contact_list = hatD_recon_motion(
        hatD, B_skel_graph, model.rep_cfg["out"], ms_dict, len(z)
    )
    return out_motion_list[0], out_contact_list[0]


##### convert all bvh to z and save as npy #####
def list_bvh_files(directory):
    bvh_files = []
    for root, dirs, files in os.walk(directory):
        if not dirs:  # leaf directory
            relative_path = os.path.relpath(root, directory)
            for file in files:
                if file.endswith(".bvh"):
                    bvh_files.append(os.path.join(relative_path, file))
    return bvh_files


import tqdm, gc


def save_bvh_z(model_epoch, bvh_dir, npy_dir):
    model, cfg, ms_dict = prepare_model_test(model_epoch, "cuda:0")
    bvh_files = list_bvh_files(bvh_dir)
    for bvh_rel_fn in tqdm.tqdm(bvh_files):
        bvh_fp = os.path.join(bvh_dir, bvh_rel_fn)
        npy_fp = os.path.join(npy_dir, bvh_rel_fn[:-4] + ".npy")
        if not os.path.exists(os.path.dirname(npy_fp)):
            os.makedirs(os.path.dirname(npy_fp))
        motion, graph_batch, z = bvh_2_graph_z(model, bvh_fp)
        # print(bvh_fp, npy_fp)
        np.save(npy_fp, z.cpu().detach().numpy())
        del motion, graph_batch, z
        gc.collect()
        torch.cuda.empty_cache()

# Scale all motion
def scale_motion(motion, unit_scale=100):
    """
    Scale motion from meters to centimeters
    - Scale skeleton offsets by unit_scale (100x)
    - Scale root positions by unit_scale
    """
    # Scale skeleton offsets
    for joint in motion.skel.joints:
        if joint.parent_joint is not None:  # Skip root joint for offset scaling
            joint.xform_from_parent_joint[:3, 3] *= unit_scale
            # Update global transforms after changing offsets
            joint.xform_global = np.dot(
                joint.parent_joint.xform_global, 
                joint.xform_from_parent_joint
            )
            # Update child joints recursively
            joint.set_xform_global_recursive(joint.xform_global)
            
            # Update body_T if it exists
            if hasattr(joint, 'body_T'):
                mid_p = 0.5 * joint.xform_from_parent_joint[:3, 3]
                z_dir_R = math.R_from_vectors(np.array([0, 0, 1]), mid_p)
                joint.body_T = conversions.Rp2T(z_dir_R, mid_p)
    
    # Scale root positions in all poses
    root_joint_idx = 0  # Assuming root is at index 0
    for pose in motion.poses:
        # Scale root position (translation part)
        pose.data[root_joint_idx][:3, 3] *= unit_scale
    
    return motion


""" ================= batch pair test (headless, all pairs in pairs_txt) ================= """


def _safe_name(rel_path):
    base = rel_path.replace("\\", "/")
    base = "__".join(base.split("/")[-2:])
    base = re.sub(r"\.npz$", "", base)
    return re.sub(r"[^A-Za-z0-9_\-\.]+", "_", base)


def read_pairs(pairs_path):
    pairs = []
    with open(pairs_path, "r", encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if ln:
                a, b = ln.split()
                pairs.append((a, b))
    return pairs


def load_pair_dataset(data_dir, src_rel, tgt_rel):
    # pair 한 쌍만 담은 독립된 PairedDataset. 항상 mi=0, ri=0 -> src, ri=1 -> tgt.
    # (pair 간 mi를 공유하면 하나의 pair가 joint-count 등으로 걸러졌을 때
    #  이후 pair들의 mi 정렬이 어긋날 수 있어, pair마다 완전히 격리시킴)
    ds = PairedDataset()
    bvh_prefix = os.path.join(os.path.dirname(data_dir), "bvh")
    ds.add_data_from_npz(
        0, os.path.join(data_dir, src_rel),
        os.path.join(bvh_prefix, str(Path(src_rel).with_suffix(""))),
    )
    ds.add_data_from_npz(
        0, os.path.join(data_dir, tgt_rel),
        os.path.join(bvh_prefix, str(Path(tgt_rel).with_suffix(""))),
    )
    return ds


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_epoch", type=str, default="ckpt0")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--data_dir", type=str, default="test/motion/processed/")
    parser.add_argument("--pairs_txt", type=str, default="pair.txt",
                       help="pair list file name under data_dir")
    parser.add_argument("--out_dir", type=str, default=None,
                       help="bvh 결과 + retarget_log.csv를 저장할 경로 (기본: result/<model_epoch>/test)")
    parser.add_argument("--unit_scale", type=float, default=100,
                       help="Scale factor for unit conversion (100 for m to cm)")
    parser.add_argument("--max_pairs", type=int, default=-1)
    parser.add_argument("--start_idx", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")

    args = parser.parse_args()

    model, cfg, ms_dict = prepare_model_test(args.model_epoch, args.device)
    out_rep_cfg = cfg["representation"]["out"]

    data_dir = os.path.join(DATA_DIR, args.data_dir)
    pairs = read_pairs(os.path.join(data_dir, args.pairs_txt))

    out_dir = args.out_dir or os.path.join(RESULT_DIR, args.model_epoch.split("/")[0], "test")
    out_dir = os.path.abspath(out_dir)
    os.makedirs(out_dir, exist_ok=True)

    log_path = os.path.join(out_dir, "retarget_log.csv")
    write_header = not os.path.exists(log_path)
    log_f = open(log_path, "a", newline="")
    log_w = csv.writer(log_f)
    if write_header:
        log_w.writerow(["idx", "status", "msg", "src_rel", "tgt_rel",
                        "src_bvh", "tgt_bvh", "out_bvh", "secs"])

    start = max(0, args.start_idx)
    end = len(pairs) if args.max_pairs <= 0 else min(len(pairs), start + args.max_pairs)
    print(f"[test] pairs={len(pairs)} | processing [{start}:{end}) | out={out_dir}")

    for idx in range(start, end):
        src_rel, tgt_rel = pairs[idx]
        t0 = time.time()
        status, msg = "OK", ""
        src_bvh_fp = tgt_bvh_fp = out_bvh_fp = ""
        try:
            pair_ds = load_pair_dataset(data_dir, src_rel, tgt_rel)
            (src_batch, tgt_batch), consq_n = get_mi_src_tgt_all_graph(
                dataset=pair_ds, mi=0, src_ri=0, tgt_ri=1, device=args.device
            )
            src_motion, tgt_motion, out_motion = retarget(
                model, src_batch, tgt_batch, ms_dict, out_rep_cfg, consq_n,
            )

            src_motion = scale_motion(src_motion, args.unit_scale)
            tgt_motion = scale_motion(tgt_motion, args.unit_scale)
            out_motion = scale_motion(out_motion, args.unit_scale)

            stem = f"pair{idx:06d}__{_safe_name(src_rel)}__TO__{_safe_name(tgt_rel)}"
            src_bvh_fp = os.path.join(out_dir, f"{stem}__SRC.bvh")
            tgt_bvh_fp = os.path.join(out_dir, f"{stem}__TGT.bvh")
            out_bvh_fp = os.path.join(out_dir, f"{stem}__OUT.bvh")
            if args.overwrite or not os.path.exists(out_bvh_fp):
                bvh.save(src_motion, src_bvh_fp)
                bvh.save(tgt_motion, tgt_bvh_fp)
                bvh.save(out_motion, out_bvh_fp)
            dt = time.time() - t0
            print(f"[{idx}] OK  {src_rel} -> {tgt_rel}  ({dt:.2f}s)")
        except Exception as e:
            import traceback
            status, msg = "FAIL", repr(e)
            dt = time.time() - t0
            print(f"[{idx}] FAIL {src_rel} -> {tgt_rel}  ({dt:.2f}s): {msg}")
            traceback.print_exc()

        log_w.writerow([idx, status, msg, src_rel, tgt_rel,
                        src_bvh_fp, tgt_bvh_fp, out_bvh_fp, f"{dt:.3f}"])
        log_f.flush()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    log_f.close()
    print(f"[test] done -> {out_dir}")
