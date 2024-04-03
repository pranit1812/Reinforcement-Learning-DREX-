from stable_baselines3.common.evaluation import evaluate_policy
from imitation.data.types import load
from imitation.algorithms.bc import BC
import gym
import numpy as np
rng = np.random.default_rng(0)

EXPERT_ID = "PPO-10"
ENV_ID = "Ant-v3"
env = gym.make(ENV_ID) # variable horizon is true for fair evaluation

rollouts = load('demonstrations/sub-optimal/'+ENV_ID+'-'+EXPERT_ID)
bc_trainer = BC(
    observation_space=env.observation_space,
    action_space=env.action_space,
    demonstrations=rollouts,
    rng=rng,
)

reward, _ = evaluate_policy(bc_trainer.policy, env, 10)
print("Avg reward before training:", reward)
bc_trainer.train(n_epochs=5)
reward, _ = evaluate_policy(bc_trainer.policy, env, 10)
print("Avg reward after training:", reward)