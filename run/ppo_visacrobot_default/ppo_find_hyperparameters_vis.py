import optuna
import gymnasium as gym
import numpy as np
import joblib

from gymnasium.wrappers import TimeLimit

from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import (
    SubprocVecEnv,
    VecNormalize,
    VecFrameStack,
    VecTransposeImage
    )

from src.model.cnn import CustomCNN
from src.wrapper.pendulum_wrapper import ResizeObservation
from src.wrapper.visual_wrapper import VisualWrapper


np.float_ = np.float64

# Objective function for Optuna
def objective(trial):
    # Create the environment
    env = gym.make('Acrobot-v1', render_mode="rgb_array")

    # Suggest hyperparameters
    learning_rate = trial.suggest_loguniform('learning_rate', 1e-5, 1e-2)
    n_steps = trial.suggest_int('n_steps', 64, 2048, log=True)
    gamma = trial.suggest_uniform('gamma', 0.9, 0.9999)
    gae_lambda = trial.suggest_uniform('gae_lambda', 0.8, 1.0)
    episode_timesteps  = trial.suggest_int('episode_timesteps', 1000, 5000)
    n_stacked_frame  = trial.suggest_int('n_stacked_frame', 4, 10)
    normalize  = trial.suggest_categorical('normalize', [True, False])
    image_height = image_width = 64
    parallel_envs = 8

    # Function to create the base environment
    def make_env(seed):
        def _init():
            env = gym.make("Acrobot-v1", render_mode="rgb_array")
            env = VisualWrapper(env)
            env = TimeLimit(env, max_episode_steps=episode_timesteps)
            env = ResizeObservation(env, (image_height, image_width))
            env.reset(seed=seed)
            return env
        return _init
    
    env = SubprocVecEnv([make_env(seed) for seed in range(parallel_envs)])
    env = VecFrameStack(env, n_stack=n_stacked_frame)
    env = VecTransposeImage(env)

    if normalize:
        env = VecNormalize(env, norm_obs=False, norm_reward=True, clip_obs=10.0)

    policy_kwargs = dict(
        features_extractor_class=CustomCNN,
        features_extractor_kwargs=dict(features_dim=256, num_frames=n_stacked_frame)  # Adjust num_frames as needed
    )

    # Create the PPO agent using the custom feature extractor
    model = PPO(
        "CnnPolicy",
        env,
        policy_kwargs=policy_kwargs,
        learning_rate=learning_rate,
        n_steps=n_steps,
        batch_size=n_steps*parallel_envs,
        gamma=gamma,
        gae_lambda=gae_lambda,
        verbose=0
    )

    # Train the model
    model.learn(total_timesteps=100000)

    # Evaluate the model
    mean_reward, _ = evaluate_policy(model, env, n_eval_episodes=10)
    
    return mean_reward


if __name__ == "__main__":
    # Create and optimize the study
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=50)

    # Print the best hyperparameters
    print("######################################")
    print("Best hyperparameters:", study.best_params)

    joblib.dump(study, "./artifacts/optuna_study_vis.pkl")
