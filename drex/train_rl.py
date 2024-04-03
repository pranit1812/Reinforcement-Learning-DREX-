from imitation.rewards.reward_wrapper import RewardVecEnvWrapper
from imitation.util.networks import RunningNorm

from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3 import PPO, SAC

import gym
import torch

import argparse
from custom_rw import SquashRewardNet

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('--env', default="HalfCheetah-v3", type=str)
    parser.add_argument('--algo', default="PPO", type=str)
    args = parser.parse_args()

    ALGO_ID = args.algo    
    algo = {"PPO": PPO, "SAC": SAC}
    ENV_ID = args.env
    # variable horizon should be disabled for sampling equal length trajectories
    # env_factory = lambda: gym.make(ENV_ID, terminate_when_unhealthy=False)
    env_factory = lambda: gym.make(ENV_ID)
    env = env_factory()

    # Load reward model
    reward_net = SquashRewardNet(
                    env.observation_space,
                    env.action_space,
                    threshold=1,
                    use_action=False, # TREX has state only reward functions
                    normalize_input_layer=RunningNorm,
                    hid_sizes=(256,256))
    reward_net.load_state_dict(torch.load('checkpoints/drex_reward_net/DREX-'+ENV_ID+'.pth'))

    # Train RL
    venv = make_vec_env(env_factory, n_envs=4)
    learned_reward_venv = RewardVecEnvWrapper(venv, reward_net.predict)
    learner = algo[ALGO_ID](policy="MlpPolicy", 
                            env=learned_reward_venv,
                            # n_steps=4096,
                            verbose=1
                            ) 
    learner.learn(1000000, callback=learned_reward_venv.make_log_callback())
    reward, _ = evaluate_policy(learner, venv, 10)
    print("Avg reward after training:", reward)
    learner.save('checkpoints/drex_policy_net/DREX-'+ENV_ID+'-'+ALGO_ID)