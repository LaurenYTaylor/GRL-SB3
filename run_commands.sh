python run_online_training.py --multirun env_name --env_name "AdroitHandHammer-v1" "CombinationLock-v1" --num_seeds 3 --algo_config/replay_buffer_kwargs/perc_guide_sampled '["cs","cs"]' '[0.5,"cs"]' [0.5,0.5]

#MAIN TESTS
python run_online_training.py --env_name "CombinationLock-v1" --num_seeds 1 --algo_config/replay_buffer_kwargs/perc_guide_sampled '[0.5,"cs"]' '[0.5,0.5]' '["cs","cs"]' --algo_config/replay_buffer_class GRLReplayBuffer --grl_config/horizon_fn agent_type --delay_training 1 0 --guide_in_actor_loss 0 1 --eval_freq 1000

#MAIN TESTS FIXED
python run_online_training.py --env_name "CombinationLock-v1" --num_seeds 10 --algo_config/replay_buffer_kwargs/perc_guide_sampled '[0.5,"cs"]' '[0.5,0.5]' '["cs","cs"]' --algo_config/replay_buffer_class GRLReplayBuffer --grl_config/horizon_fn agent_type --grl_config/delay_training False --grl_config/guide_in_actor_loss True False --eval_freq 2500 --algo_config/buffer_size 100000

python run_online_training.py --env_name "AdroitHandHammer-v1" --num_seeds 3 --algo_config/replay_buffer_kwargs/perc_guide_sampled '[0.5,"cs"]' '[0.5,0.5]' '["cs","cs"]' --algo_config/replay_buffer_class GRLReplayBuffer --grl_config/horizon_fn agent_type --grl_config/delay_training True False --grl_config/guide_in_actor_loss True False --eval_freq 10000 --total_timesteps 1000000