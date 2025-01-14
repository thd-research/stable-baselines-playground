from stable_baselines3.ppo import PPO
from stable_baselines3.common.on_policy_algorithm import OnPolicyAlgorithm
from PIL import Image
import matplotlib.pyplot as plt

class DebugPPO(PPO):
    def collect_rollouts(self, env, callback, rollout_buffer, n_rollout_steps):
        result = super().collect_rollouts(env, callback, rollout_buffer, n_rollout_steps)
        # # Debug: Check the rollout buffer observations
        # fig, ax = plt.subplots(10, 1, figsize=(5, 10))
        # for i in range(10):
        #     print(f"[ {i} ] Rollout buffer first observation stats: "
        #           f"Min={rollout_buffer.observations[i].min()}, "
        #           f"Max={rollout_buffer.observations[i].max()}, "
        #           f"Mean={rollout_buffer.observations[i].mean()}, "
        #           f"Shape={rollout_buffer.observations[i].shape}")
            
        #     image = rollout_buffer.observations[i].transpose(0, 2, 3, 1)[0, :, :, :3].astype(int)
            
        #     ax[i].imshow(image)
        #     ax[i].set_title(f"Image {i}")
        #     ax[i].axis("off")

        # plt.tight_layout()
        # plt.show(block=False)
        return result
