# python metric/metric.py --result_dir result/260803_cfg_VT_fold0/test --gt_dir result//test --pairs_txt  data/Trueboness_processed_byVT/processed/truebones_vt_exact_test.txt 
python metric/metric.py --result_dir result/260803_cfg_VT_fold0/test --gt_dir data/Trueboness_processed_byVT/augmented --pairs_txt  data/Trueboness_processed_byVT/processed/truebones_vt_groups_fold0_test.txt

# --out_csv result/260803_cfg_VT_split/test/metrics_test.csv