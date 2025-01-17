from gymnasium import (
    Wrapper,
    spaces  # Import spaces to define the observation space
)
import numpy as np


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