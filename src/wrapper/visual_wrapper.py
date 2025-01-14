from gymnasium import Wrapper


class VisualWrapper(Wrapper):
    def __init__(self, env):
        super().__init__(env)

    def step(self, action):
        # Step in the environment
        _, reward, done, truncated, info = super().step(action)

        image = self.render()
        return image, reward, done, truncated, info

    
    def reset(self, *, seed = None, options = None):
        _, info = super().reset(seed=seed, options=options)

        # Render for image-based observations
        image = self.render()
        return image, info
    
    def render(self):
        # Avoid creating rendering resources during initialization
        return super().render()
    
    def close(self):
        # Ensure resources are properly cleaned up
        if hasattr(self, "screen") and self.screen is not None:
            self.screen = None
        super().close()