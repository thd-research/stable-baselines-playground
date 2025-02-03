from gymnasium import (
    Wrapper,
    RewardWrapper,
    spaces  # Import spaces to define the observation space
)
import numpy as np

from gymnasium import ObservationWrapper
import gymnasium as gym



class VisualWrapper(Wrapper):
    def __init__(self, env):
        super().__init__(env)
        image_shape = (500, 500, 3)
        self.observation_space = spaces.Box(low=0, high=255, shape=image_shape, dtype=np.uint8)

    def step(self, action):
        # Step in the environment
        # _, reward, done, truncated, info = super().step(action)
        _, reward, done, truncated, info = super().step(action if action.dtype != np.float32 else action.astype(np.float64))

        image = self.render()
        return image, reward, done, truncated, info
    
    def reset(self, *, seed = None, options = None):
        _, info = super().reset(seed=seed, options=options)

        # Render for image-based observations
        image = self.render()
        return image, info
    
    def render(self):
        # Assign last_u a None to prevend from drawing torque arrows
        self.env.last_u = None
        # Avoid creating rendering resources during initialization
        return super().render()
    
    def close(self):
        # Ensure resources are properly cleaned up
        if hasattr(self, "screen") and self.screen is not None:
            self.screen = None
        super().close()

class LunarLanderRewardEngineering(Wrapper):
    def step(self, action):
        # Step in the environment
        # _, reward, done, truncated, info = super().step(action)
        image_obs, reward, done, truncated, info = super().step(action if action.dtype != np.float32 else action.astype(np.float64))

        if image_obs.shape[1] == 200 and image_obs.shape[2] == 80:
            count = np.count_nonzero(np.isclose(self.init_img, image_obs, atol=7))
            if 47760 < count < 47660:
                reward = -1

        return image_obs, reward, done, truncated, info
    
    def reset(self, *, seed = None, options = None):
        self.init_img, info = super().reset(seed=seed, options=options)

        return self.init_img.copy(), info
    

class CropObservation(ObservationWrapper):
    def __init__(self, env, shape=(500, 500), width_center=(100)):
        super(CropObservation, self).__init__(env)
        self.shape = shape
        self.width_center = width_center
        self.left_edge = int(self.shape[1]/2) - self.width_center
        self.right_edge = int(self.shape[1]/2) + self.width_center
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(shape[0], int(width_center*2), 3), dtype=np.uint8
        )

    def observation(self, observation):
        # Debug: Check if the observation is empty or not
        if observation is None or observation.size == 0:
            print("Error: Observation is empty or not properly generated.")
            raise ValueError("Observation is empty or not properly generated.")

        # Resize the observation using OpenCV
        cropped_observation = observation[:, self.left_edge:self.right_edge, :]

        return cropped_observation
