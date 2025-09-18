"""
Process a paired motion dataset(.bvh files) to .npz files.
    python preprocess/preprocess_data.py --train  --data "train" --wopair  

'/home/inseo/Github/SAME/src/../data/sample/motion/bvh'
"""
import os, argparse, pathlib, torch
import numpy as np
import copy 

from mypath import *
from fairmotion.data import bvh
from utils.motion_utils import motion_normalize_h2s
from conversions.motion_to_graph import motion_2_states

alternative_map = {
    "ToSpine": "LowerBack",
    "LHipJoint": "LeftHipJoint",
    "RHipJoint": "RightHipJoint",
    "LeftToe": "LeftToeBase",
    "RightToe": "RightToeBase",
    "LeftToe_End": "LeftToeBase_End",
    "RightToe_End": "RightToeBase_End",
    
    # Hamster (57 joints)
    # End-Effectors
    "Bip01_R_Toe0Nub": "RightToeBase_End",
    "Bip01_L_Toe0Nub": "LeftToeBase_End",
    "Bip01_HeadNub": "Head_End", 
    "Bip01_R_Finger0Nub": "RightHand_End",
    "Bip01_L_Finger0Nub": "LeftHand_End",
}

# Append Tpose
def create_tpose_frame(motion):
    """
    Create T-pose frame from the motion's skeleton structure.
    Returns a motion object with a single T-pose frame.
    """
    # T-pose는 보통 모든 조인트를 기본 rotation(identity)으로 설정
    # 첫 번째 프레임을 기반으로 T-pose 생성
    tpose_motion = copy.deepcopy(motion) # .copy()
    
    # 모든 프레임을 1개로 줄이고 T-pose로 설정
    tpose_motion.poses = [copy.deepcopy(motion.poses[0])] # .copy() 첫 번째 프레임을 복사
    
    # 모든 조인트의 rotation을 identity로 설정 (T-pose)
    for jid, joint_name in enumerate(tpose_motion.skel.joints):
        joint = tpose_motion.skel.joints[jid]
        if joint != motion.skel.root_joint:  # root joint는 제외
            # Identity quaternion [1, 0, 0, 0] 또는 Euler [0, 0, 0]
            tpose_motion.poses[0].data[jid, :] = 0.0
    
    return tpose_motion


def save_tpose_separately(motion, character_name, output_dir_path):
    """
    Save T-pose as a separate file for the character.
    """
    # T-pose 생성
    tpose_motion = create_tpose_frame(motion)
    
    # T-pose를 skeleton state와 pose state로 변환
    skel_state, poses_state = motion_2_states(tpose_motion)
    lo, go, qb, edges = skel_state
    q, p, r, pv, qv, pprev, c = poses_state
    
    # T-pose 저장 경로 설정
    tpose_dir = os.path.join(output_dir_path, character_name)
    if not os.path.exists(tpose_dir):
        os.makedirs(tpose_dir)
    
    tpose_save_path = os.path.join(tpose_dir, "__TPOSE.npz")
    
    # T-pose 저장
    np.savez_compressed(
        tpose_save_path,
        lo=lo,
        go=go,
        qb=qb,
        edges=edges,
        q=q,
        p=p,
        r=r,
        pv=pv,
        qv=qv,
        pprev=pprev,
        c=c,
    )
    
    print(f"T-pose saved for {character_name}: {tpose_save_path}")
    return tpose_save_path


def prepend_tpose_to_motion(motion, tpose_motion):
    """
    Prepend T-pose frame to the beginning of the motion.
    """
    # 원본 모션에 T-pose를 첫 번째 프레임으로 추가
    combined_motion = copy.deepcopy(motion) # .copy()
    
    # T-pose 프레임을 맨 앞에 추가
    tpose_frame = tpose_motion.poses[0]
    combined_motion.poses = [tpose_frame] + combined_motion.poses
    
    return combined_motion
# end 

def preprocess_motion(motion, save_path, normalized=False, add_tpose=True, scale_factor=100):
    if not normalized:
        motion, tpose = motion_normalize_h2s(motion, alternative_map, False)  # 0.2~3s
    
    # Apply unit scaling before processing
    if scale_factor != 1.0:
        motion = apply_unit_scaling(motion, scale_factor)
        print(f"Applied unit scaling factor: {scale_factor}")
    
    if add_tpose:
        # 캐릭터 이름 추출 (파일 경로에서)
        character_name = os.path.basename(os.path.dirname(save_path))
        output_dir_path = os.path.dirname(os.path.dirname(save_path))
        
        # T-pose 생성 및 별도 저장
        tpose_motion = create_tpose_frame(motion)
        tpose_save_path = save_tpose_separately(tpose_motion, character_name, output_dir_path)
        
        # 원본 모션에 T-pose를 첫 번째 프레임으로 추가
        motion_with_tpose = prepend_tpose_to_motion(motion, tpose_motion)
        
        # T-pose가 포함된 모션을 처리
        skel_state, poses_state = motion_2_states(motion_with_tpose)
    else:
        # 기존 방식: T-pose 없이 처리
        skel_state, poses_state = motion_2_states(motion)

    # Additional scaling for processed data (double scaling prevention check)
    # skel_state, poses_state = scale_processed_data(skel_state, poses_state, scale_factor)

    lo, go, qb, edges = skel_state
    # In [4]: for ss in skel_state: print(ss.shape)
    # (28, 3)
    # (28, 3)
    # (28,)
    # (E, 2+ number of features(=currently 2: depth, reverse_depth))

    q, p, r, pv, qv, pprev, c = poses_state
    # In [5]: for ps in poses_state: print(ps.shape)
    # (300, 28, 6)
    # (300, 28, 3)
    # (300, 4)
    # (300, 28, 3)
    # (300, 28, 6)
    # (300, 28, 3)
    # (300, 28)

    save_dir = os.path.dirname(save_path)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    np.savez_compressed(
        save_path,
        lo=lo,
        go=go,
        qb=qb,
        edges=edges,
        q=q,
        p=p,
        r=r,
        pv=pv,
        qv=qv,
        pprev=pprev,
        c=c,
    )
    print("saved at ", save_path)


def preprocess_single_data(
    input_dir_path,
    output_dir_path,
    append_log=False,
    scale_factor=100,
):
    if not os.path.exists(output_dir_path):
        os.makedirs(output_dir_path)
    output_pair_path = os.path.join(output_dir_path, "pair.txt")
    valid_cnt = 0

    # bvh to npy 
    all_files_recursive = list(pathlib.Path(input_dir_path).rglob("*"))
    with open(output_pair_path, "a" if append_log else "w") as output_file:
        for filepath in all_files_recursive:
            filepath = str(filepath)
            if filepath.endswith(".bvh"):
                relpath = os.path.relpath(filepath, input_dir_path)
                relpath_npz = relpath[:-4] + ".npz"
                out_save_full_path = os.path.join(output_dir_path, relpath_npz)

                if os.path.exists(out_save_full_path):
                    continue  # already processed
                valid_cnt += 1

                motion = bvh.load(filepath, ignore_root_skel=True, ee_as_joint=True)
                # preprocess_motion(motion, out_save_full_path, normalized=False)
                preprocess_motion(motion, out_save_full_path, normalized=False, 
                                add_tpose=True, scale_factor=scale_factor)

                outpaths = [relpath_npz, relpath_npz]
                output_file.write("\t".join(outpaths) + "\n")

def preprocess_paired_data(
    input_dir_path,
    output_dir_path,
    append_log=False,  # True when continuing the process (killed unexpectedly) or when merging multiple dataset
):
    data_dir = os.path.relpath(input_dir_path, DATA_DIR)
    input_pair_path = os.path.join(input_dir_path, "pair.txt")

    if not os.path.exists(output_dir_path):
        os.makedirs(output_dir_path)
    output_pair_path = os.path.join(output_dir_path, "pair.txt")

    valid_cnt = 0
    with open(input_pair_path, "r") as file:
        with open(output_pair_path, "a" if append_log else "w") as output_file:
            for line in file:
                words = line.rstrip().split(", ")

                batch_i, skel_i, relpath_a, relpath_b = words

                relpaths = [
                    os.path.relpath(relpath_a, data_dir),
                    os.path.relpath(relpath_b, data_dir),
                ]
                sanity_check = [
                    os.path.exists(os.path.join(input_dir_path, relpath_i))
                    for relpath_i in relpaths
                ]
                if not all(sanity_check):
                    for rp, sc in zip(relpaths, sanity_check):
                        if not sc:
                            print(
                                "err: no such file: ", os.path.join(input_dir_path, rp)
                            )
                    continue

                outpaths = []
                for relpath_i in relpaths:
                    in_motion_full_path = os.path.join(input_dir_path, relpath_i)
                    out_save_full_path = os.path.join(output_dir_path, relpath_i)
                    outpaths.append(relpath_i + ".npz")

                    if os.path.exists(out_save_full_path + ".npz"):
                        continue  # already processed
                    valid_cnt += 1

                    motion = bvh.load(
                        in_motion_full_path, ignore_root_skel=True, ee_as_joint=True
                    )
                    preprocess_motion(motion, out_save_full_path, normalized=False)

                # if there's any error while processing path_a or path_b it won't be written to output_file
                output_file.write("\t".join(outpaths) + "\n")

                # # for testing purpose
                # if valid_cnt >= 90:
                #     break


def compute_statistics(
    output_dir,
):
    pair_path = os.path.join(DATA_DIR, output_dir, "pair.txt")
    assert os.path.exists(pair_path)

    rel_paths = []
    with open(pair_path, "r") as pair_file:
        for line in pair_file:
            if line.strip() == "":
                continue
            rel_paths.extend(line.strip().split())  # src_rel_path, dst_rel_path

    npz_unique_paths = [
        os.path.join(DATA_DIR, output_dir, rel_path) for rel_path in set(rel_paths)
    ]
    npz_unique_vals = [np.load(npz_path) for npz_path in npz_unique_paths]
    normalize_keys = ["lo", "go", "q", "p", "r", "pv", "qv", "pprev"]

    mean_stds_dict = {}
    for key in normalize_keys:
        key_dim = npz_unique_vals[0][key].shape[-1]
        v_stack = np.vstack(
            [vals[key].reshape(-1, key_dim) for vals in npz_unique_vals]
        )
        mean_stds_dict[key + "_m"] = torch.Tensor(v_stack.mean(axis=0))
        mean_stds_dict[key + "_s"] = torch.Tensor(v_stack.std(axis=0))

    ms_dict_path = os.path.join(DATA_DIR, output_dir, "ms_dict.pt")
    torch.save(mean_stds_dict, ms_dict_path)
    print("saved statistics(mean/std) of", len(rel_paths), "files into", ms_dict_path)


def apply_unit_scaling(motion, scale_factor=100):
    """
    Apply unit scaling to motion data (m to cm by default).
    Scales position data but keeps rotation data unchanged.
    """
    if scale_factor == 1.0:
        return motion
    
    print(f"Applying unit scaling with factor: {scale_factor}")
    
    # Scale skeleton positions
    for joint in motion.skel.joints:
        # Scale joint positions (local offsets)
        if hasattr(joint, 'offset') and joint.offset is not None:
            joint.offset = joint.offset * scale_factor
        
        # Scale transformation matrices if they exist
        if hasattr(joint, 'xform_from_parent_joint') and joint.xform_from_parent_joint is not None:
            # Only scale translation part (last column, first 3 rows)
            joint.xform_from_parent_joint[:3, 3] *= scale_factor
        
        if hasattr(joint, 'xform_global') and joint.xform_global is not None:
            # Only scale translation part
            joint.xform_global[:3, 3] *= scale_factor
    
    # Scale pose data (root translations)
    for pose in motion.poses:
        if hasattr(pose, 'data') and pose.data is not None:
            # Typically, root joint (index 0) contains translation
            # Other joints contain rotation data which should not be scaled
            root_translation_indices = slice(0, 3)  # Usually first 3 values are root translation
            pose.data[0, root_translation_indices] *= scale_factor
        
        # If pose has transformation matrices
        elif hasattr(pose, 'transform') and pose.transform is not None:
            pose.transform[:3, 3] *= scale_factor
    
    return motion

# def scale_processed_data(skel_state, poses_state, scale_factor=100):
#     """
#     Scale the processed data arrays for unit conversion.
#     """
#     if scale_factor == 1.0:
#         return skel_state, poses_state
    
#     lo, go, qb, edges = skel_state
#     q, p, r, pv, qv, pprev, c = poses_state
    
#     # Scale position-related data
#     lo_scaled = lo * scale_factor  # local offsets
#     go_scaled = go * scale_factor  # global positions
#     p_scaled = p * scale_factor    # positions
#     pv_scaled = pv * scale_factor  # position velocities
#     pprev_scaled = pprev * scale_factor  # previous positions
    
#     # Root position in 'r' might also need scaling
#     r_scaled = r.copy()
#     if r.shape[-1] >= 3:  # If r contains position data
#         r_scaled[..., :3] *= scale_factor
    
#     # Quaternion data (q, qv) and contact data (c) don't need scaling
#     # Boolean data (qb) and edge data (edges) don't need scaling
    
#     skel_state_scaled = (lo_scaled, go_scaled, qb, edges)
#     poses_state_scaled = (q, p_scaled, r_scaled, pv_scaled, qv, pprev_scaled, c)
    
#     return skel_state_scaled, poses_state_scaled

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="sample")
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--wopair", action="store_true")
    parser.add_argument("--append_log", action="store_true")
    parser.add_argument("--unit_scale", type=float, default=100, 
                       help="Scale factor for unit conversion (100 for m to cm, 1 for no scaling)")
    args = parser.parse_args()

    in_path = os.path.join(DATA_DIR, args.data, "motion", "bvh")
    
    # Add scale suffix to output directory if scaling is applied
    if args.unit_scale != 1.0:
        out_path = os.path.join(DATA_DIR, args.data, "motion", "processed")
        # out_path = os.path.join(DATA_DIR, args.data, "motion", f"processed_scaled_{int(args.unit_scale)}")
        print(f"Using scaled output directory: {out_path}")
    else:
        out_path = os.path.join(DATA_DIR, args.data, "motion", "processed")

    if args.wopair:
        preprocess_ftn = lambda inp, outp, append: preprocess_single_data(inp, outp, append, args.unit_scale)
    else:
        preprocess_ftn = lambda inp, outp, append: preprocess_paired_data(inp, outp, append, args.unit_scale)

    # if args.wopair:
    #     preprocess_ftn = preprocess_single_data
    # else:
    #     preprocess_ftn = preprocess_paired_data # 

    preprocess_ftn(in_path, out_path, args.append_log)

    if args.train:
        compute_statistics(out_path)
