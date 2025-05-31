# python run_online_training.py --env_name adroit --horizon_fn all --num_seeds 1 --guide_in_buffer 1
# python run_online_training.py --env_name extra --horizon_fn all --num_seeds 4 --guide_in_buffer 1
# python run_online_training.py --env_name adroit --sample_perc 0.1 --horizon_fn reward_var --num_seeds 4 --guide_in_buffer 1
# python run_online_training.py --env_name extra --sample_perc 0.1 --horizon_fn reward_var --num_seeds 4 --guide_in_buffer 1
# python run_online_training.py --env_name adroit --sample_perc 0.25 --horizon_fn reward_var --num_seeds 4 --guide_in_buffer 1
# python run_online_training.py --env_name extra --sample_perc 0.25 --horizon_fn reward_var --num_seeds 4 --guide_in_buffer 1
python run_online_training.py --env_name all --horizon_fn all --num_seeds 1 --guide_in_buffer 1
#python run_online_training.py --env_name extra --horizon_fn all --num_seeds 1 --guide_in_buffer 1
