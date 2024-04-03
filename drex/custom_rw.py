from imitation.rewards.reward_nets import BasicRewardNet
from imitation.util.networks import RunningNorm
import torch as th
import gym

class SquashRewardNet(BasicRewardNet):
    def __init__(
            self,
            observation_space: gym.Space,
            action_space: gym.Space,
            threshold: float,
            use_state: bool = True,
            use_action: bool = True,
            use_next_state: bool = False,
            use_done: bool = False,
            **kwargs,
    ):
        super().__init__(observation_space, action_space, 
                         use_state, use_action, use_next_state, use_done, **kwargs)
        self.threshold = threshold
    
    def forward(self, state, action, next_state, done):
        outputs = super().forward(state, action, next_state, done)

        # new_outputs = th.clip(outputs, min=-self.threshold, max=self.threshold)
        
        # low, high = -self.threshold, self.threshold
        # new_outputs = 2.0 * ((outputs - low) / (high - low)) - 1.0

        new_outputs = 1*th.tanh(outputs)

        return new_outputs

def get_ensemble_members(net_class, num_models, env, use_action=False):
    reward_members = [net_class(
                        env.observation_space,
                        env.action_space,
                        threshold=1,
                        use_action=use_action, # TREX has state only reward functions
                        normalize_input_layer=RunningNorm,
                        hid_sizes=(256,256))
                        for _ in range(num_models)]
    return reward_members