# bash run_eval.sh

# test 
python src/same/test.py --data_dir "Trueboness_processed_byVT/processed/" --model_epoch "260803_cfg_VT_fold0" --pairs_txt "truebones_vt_groups_fold0_test.txt"
python src/same/test.py --data_dir "Trueboness_processed_byVT/processed/" --model_epoch "260803_cfg_VT_fold1" --pairs_txt "truebones_vt_groups_fold1_test.txt"
python src/same/test.py --data_dir "Trueboness_processed_byVT/processed/" --model_epoch "260803_cfg_VT_fold2" --pairs_txt "truebones_vt_groups_fold2_test.txt"

# eval
python metric/metric.py --result_dir result/260803_cfg_VT_fold0/test --gt_dir data/Trueboness_processed_byVT/augmented --pairs_txt  data/Trueboness_processed_byVT/processed/truebones_vt_groups_fold0_test.txt
python metric/metric.py --result_dir result/260803_cfg_VT_fold1/test --gt_dir data/Trueboness_processed_byVT/augmented --pairs_txt  data/Trueboness_processed_byVT/processed/truebones_vt_groups_fold1_test.txt
python metric/metric.py --result_dir result/260803_cfg_VT_fold2/test --gt_dir data/Trueboness_processed_byVT/augmented --pairs_txt  data/Trueboness_processed_byVT/processed/truebones_vt_groups_fold2_test.txt
