import gym
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.env_util import make_vec_env

ALGO_ID = "PPO"
algo = {"PPO": PPO, "SAC": SAC}

ENV_ID = "Ant-v3"
env_factory = lambda: gym.make(ENV_ID) # variable horizon is important for training the expert

venv = make_vec_env(env_factory, n_envs=4)
expert = algo[ALGO_ID]("MlpPolicy", venv, verbose=1)

expert.learn(total_timesteps=1000000)
reward, _ = evaluate_policy(expert, venv, 10)
print("Avg reward after training:", reward)

expert.save('checkpoints/expert_policies/'+ENV_ID+'-'+ALGO_ID)

# optimality = "sub-optimal/"
# # optimality = ""
# expert = algo[ALGO_ID].load('checkpoints/expert_policies/'+optimality+ENV_ID+'-'+ALGO_ID)

from imitation.data.types import save
from imitation.data import rollout
from imitation.data.wrappers import RolloutInfoWrapper
from stable_baselines3.common.vec_env import DummyVecEnv
import numpy as np

NUM_EPISODES = 20
rollouts = rollout.rollout(
    expert,
    DummyVecEnv([lambda: RolloutInfoWrapper(env_factory())]),
    rollout.make_sample_until(min_timesteps=None, min_episodes=NUM_EPISODES),
    rng=np.random.default_rng(0),
)

save('demonstrations/'+ENV_ID+'-'+ALGO_ID+'-'+str(NUM_EPISODES), rollouts)
