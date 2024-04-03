from stable_baselines3 import PPO, SAC
from stable_baselines3.common.evaluation import evaluate_policy
import gym
import time
import numpy as np

ALGO_ID = "PPO"
algo = {"PPO": PPO, "SAC": SAC}

ENV_ID = "HalfCheetah-v3"
# optimality = "sub-optimal/"
optimality = ""
expert = algo[ALGO_ID].load('checkpoints/drex_policy_net/DREX-'+optimality+ENV_ID+'-'+ALGO_ID)

# variable horizon should be disabled for sampling equal length trajectories
# env_factory = lambda: gym.make(ENV_ID, terminate_when_unhealthy=False)
env_factory = lambda: gym.make(ENV_ID)
env = env_factory()

# expert = algo[ALGO_ID]('MlpPolicy', env)
reward, _ = evaluate_policy(expert, env, 10)
print("Avg reward:", reward)

for i in range(5):
    obs = env.reset()
    done = False
    episode_reward = 0
    act_square_sum = 0
    while not done:
        action, _ = expert.predict(obs, deterministic=True)
        # action = env.action_space.sample()
        act_square_sum += np.sum(np.squeeze(action)**2)
        obs, reward, done, info = env.step(action)
        episode_reward += reward
        env.render()
        time.sleep(0.01)
    print('Episode reward: ', episode_reward)
    print('Act: ', act_square_sum)