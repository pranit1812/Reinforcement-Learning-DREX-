from imitation.data import rollout
from imitation.algorithms.base import BaseImitationAlgorithm
from imitation.data.wrappers import RolloutInfoWrapper
from imitation.data.types import TrajectoryWithRew
from imitation.algorithms.preference_comparisons import PreferenceDataset

from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise

import numpy as np


class NoiseInjectedPolicyWrapper(BasePolicy):
    def __init__(self, policy, action_noise_type, noise_level):
        super().__init__(policy.observation_space, policy.action_space)
        self.policy = policy
        self.action_noise_type = action_noise_type

        if action_noise_type == 'normal':
            mu, std = np.zeros(self.action_space.shape), noise_level * np.ones(self.action_space.shape)
            self.action_noise = NormalActionNoise(mean=mu, sigma=std)
        elif action_noise_type == 'ou':
            mu, std = np.zeros(self.action_space.shape), noise_level * np.ones(self.action_space.shape)
            self.action_noise = OrnsteinUhlenbeckActionNoise(mean=mu, sigma=std)
        elif action_noise_type == 'epsilon':
            self.epsilon = noise_level
        else:
            assert False, "no such action noise type: %s" % (action_noise_type)
    
    def _predict(self, observation, deterministic = False): # copied from sb3 templates
        return self.get_distribution(observation).get_actions(deterministic=deterministic)

    def predict(self, observation, state = None, episode_start = None, deterministic = False):
        if self.action_noise_type == 'epsilon':
            if np.random.random() < self.epsilon:
                action = np.expand_dims(self.action_space.sample(), 0)
            else:
                action, _ = self.policy.predict(observation, deterministic)
        else:
            action, _ = self.policy.predict(observation, deterministic)
            action += self.action_noise()

        return np.clip(action, self.action_space.low, self.action_space.high), state

    def reset_noise(self):
        self.action_noise.reset()


class DREX(BaseImitationAlgorithm):
    def __init__(self, 
                 expert,
                 reward_trainer,
                 env_factory,
                 n_noise_levels,
                 k,
                 n_pairs,
                 noise_pref_gap,
                 fragment_len,
                 rng
            ):
        
        super().__init__(
            custom_logger=None,
            allow_variable_horizon=False,
        )

        self.n_pairs = n_pairs
        self.fragment_len = fragment_len
        self.noise_pref_gap = noise_pref_gap

        self.reward_trainer = reward_trainer
        self.reward_trainer.logger = self.logger
        
        noise_schedule = np.linspace(0,1,n_noise_levels)
        self.ranked_trajectories = self.generate_ranked_trajectories(noise_schedule, k, env_factory(), 
                                                        expert, action_noise_type='epsilon',
                                                        rng=rng)
        self.log_rankings(self.ranked_trajectories)        
        self.dataset = PreferenceDataset(max_size=n_pairs) # allow finite queue size for flusing every epoch

    def generate_ranked_trajectories(self, noise_schedule, k, env, expert, action_noise_type, rng):
        self.logger.log(f"Generating ranked trajectories with schedule {noise_schedule}")
        ranked_trajectories = {}
        for noise_level in noise_schedule:
            rollouts = rollout.rollout(
                NoiseInjectedPolicyWrapper(policy=expert, action_noise_type=action_noise_type, noise_level=noise_level),
                DummyVecEnv([lambda: RolloutInfoWrapper(env)]),
                rollout.make_sample_until(min_timesteps=None, min_episodes=k),
                rng=rng,
            )
            ranked_trajectories[noise_level] = rollouts
        
        return ranked_trajectories

    def generate_fragments_and_preferences(self, ranked_trajectories, n_pairs, fragment_len, noise_pref_gap):
        self.logger.log(f"Generated {n_pairs} fragment pairs of length {fragment_len} and preferences")
        noise_schedule = list(ranked_trajectories.keys())
        fragments = []
        preferences = np.zeros((n_pairs,), dtype=np.float32)
        for i in range(n_pairs):
            idx_1, idx_2 = np.random.choice(len(noise_schedule), size=2, replace=False)
            while abs(noise_schedule[idx_1]-noise_schedule[idx_2])<noise_pref_gap:
                idx_1, idx_2 = np.random.choice(len(noise_schedule), size=2, replace=False)
            
            # less noise has higher preference
            if noise_schedule[idx_1] < noise_schedule[idx_2]:
                preferences[i] = 1.0
            
            trajectories_1 = ranked_trajectories[noise_schedule[idx_1]]
            trajectories_2 = ranked_trajectories[noise_schedule[idx_2]]
            idx_1, idx_2 = np.random.choice(len(trajectories_1)), np.random.choice(len(trajectories_2))
            trajectory_1, trajectory_2 = trajectories_1[idx_1], trajectories_2[idx_2]
            assert len(trajectory_1) > fragment_len and len(trajectory_2) > fragment_len
            idx_1, idx_2 = np.random.choice(len(trajectory_1)-fragment_len-1), np.random.choice(len(trajectory_2)-fragment_len-1)
            fragments.append((
                TrajectoryWithRew(trajectory_1.obs[idx_1:idx_1+fragment_len+1,:], 
                                trajectory_1.acts[idx_1:idx_1+fragment_len,:],
                                trajectory_1.infos,
                                trajectory_1.terminal,
                                trajectory_1.rews[idx_1:idx_1+fragment_len]),
                TrajectoryWithRew(trajectory_2.obs[idx_2:idx_2+fragment_len+1,:], 
                                trajectory_2.acts[idx_2:idx_2+fragment_len,:],
                                trajectory_2.infos,
                                trajectory_2.terminal,
                                trajectory_2.rews[idx_2:idx_2+fragment_len])                              
                ))

        return fragments, preferences

    def log_rankings(self, ranked_trajectories):
        for noise_level in ranked_trajectories:
            rewards = []
            samples = 0
            rollouts = ranked_trajectories[noise_level]
            for roll in rollouts:
                rewards.append(np.sum(roll.rews))
                samples += roll.obs.shape[0]

            self.logger.log(f"Noise: {noise_level}")
            self.logger.log(f"#Samples: {samples}")
            self.logger.log(f"Best: {max(rewards)}")
            self.logger.log(f"Worst: {min(rewards)}")
            self.logger.log(f"Avg: {np.mean(np.array(rewards))}")
            self.logger.log(f"-----------------------------------")

    def train(self, n_epochs):
        reward_loss, reward_accuracy = None, None
        for e in range(n_epochs):
            self.logger.log(f"Starting epoch {e}")
            fragments_batch, preferences_batch = self.generate_fragments_and_preferences(self.ranked_trajectories, 
                                        self.n_pairs,
                                        self.fragment_len,
                                        self.noise_pref_gap)
            self.dataset.push(fragments_batch, preferences_batch) # should evict previous batch as it acts like a deque
            self.reward_trainer.train(self.dataset)

            base_key = self.logger.get_accumulate_prefixes() + "reward/final/train"
            assert f"{base_key}/loss" in self.logger.name_to_value
            assert f"{base_key}/accuracy" in self.logger.name_to_value
            reward_loss = self.logger.name_to_value[f"{base_key}/loss"]
            reward_accuracy = self.logger.name_to_value[f"{base_key}/accuracy"]
            self.logger.dump(e+1)
        
        return reward_loss, reward_accuracy
