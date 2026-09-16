# bash run_eval.sh

# test 
# python src/same/test.py --data_dir "Trueboness_processed_byVT/processed/" --model_epoch "260803_cfg_VT_fold0" --pairs_txt "truebones_vt_groups_fold0_test.txt"
# python src/same/test.py --data_dir "Trueboness_processed_byVT/processed/" --model_epoch "260803_cfg_VT_fold1" --pairs_txt "truebones_vt_groups_fold1_test.txt"
# python src/same/test.py --data_dir "Trueboness_processed_byVT/processed/" --model_epoch "260803_cfg_VT_fold2" --pairs_txt "truebones_vt_groups_fold2_test.txt"

# # eval
# python metric/metric.py --result_dir result/260803_cfg_VT_fold0/test --gt_dir data/Trueboness_processed_byVT/augmented --pairs_txt  data/Trueboness_processed_byVT/processed/truebones_vt_groups_fold0_test.txt
# python metric/metric.py --result_dir result/260803_cfg_VT_fold1/test --gt_dir data/Trueboness_processed_byVT/augmented --pairs_txt  data/Trueboness_processed_byVT/processed/truebones_vt_groups_fold1_test.txt
# python metric/metric.py --result_dir result/260803_cfg_VT_fold2/test --gt_dir data/Trueboness_processed_byVT/augmented --pairs_txt  data/Trueboness_processed_byVT/processed/truebones_vt_groups_fold2_test.txt

# recon (A->A) + cycle (A->B->A'): source motions from pair.txt only, target = species/skeleton only.
# GT = original source (same skeleton), so mpjpe/rot_err are valid without a retarget GT.
python src/eval_recon_cycle.py --model_epoch 260803_cfg_VT_fold0 --data_dir "Trueboness_processed_byVT/processed/" --pairs_txt truebones_vt_groups_fold0_test.txt --out_csv result/260803_cfg_VT_fold0/test/recon_cycle.csv
python src/eval_recon_cycle.py --model_epoch 260803_cfg_VT_fold1 --data_dir "Trueboness_processed_byVT/processed/" --pairs_txt truebones_vt_groups_fold1_test.txt --out_csv result/260803_cfg_VT_fold1/test/recon_cycle.csv
python src/eval_recon_cycle.py --model_epoch 260803_cfg_VT_fold2 --data_dir "Trueboness_processed_byVT/processed/" --pairs_txt truebones_vt_groups_fold2_test.txt --out_csv result/260803_cfg_VT_fold2/test/recon_cycle.csv

# metrics: BVH metrics (jerk/foot_skating/ground_pen/freq_align/contact_con) + merge recon_cycle.csv.
# No target-GT metrics (pair target is not real GT); source metrics use --src_dir.
python metric/metric.py --result_dir result/260803_cfg_VT_fold0/test --pairs_txt data/Trueboness_processed_byVT/processed/truebones_vt_groups_fold0_test.txt --src_dir data/Trueboness_processed_byVT/augmented
python metric/metric.py --result_dir result/260803_cfg_VT_fold1/test --pairs_txt data/Trueboness_processed_byVT/processed/truebones_vt_groups_fold1_test.txt --src_dir data/Trueboness_processed_byVT/augmented
python metric/metric.py --result_dir result/260803_cfg_VT_fold2/test --pairs_txt data/Trueboness_processed_byVT/processed/truebones_vt_groups_fold2_test.txt --src_dir data/Trueboness_processed_byVT/augmented
