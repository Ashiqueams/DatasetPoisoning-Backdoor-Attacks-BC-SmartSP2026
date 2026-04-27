from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import VecFrameStack

seed = 1

env_kwargs = dict(continuous=True)
env = make_vec_env("CarRacing-v3", env_kwargs=env_kwargs, n_envs=8, seed=seed)
# env = VecFrameStack(env, n_stack=4) # stacking 4 frames 
model = PPO(
        policy="CnnPolicy",
        env=env,
        learning_rate=3e-4,        # reduced from 1e-3
        n_steps=2048,
        batch_size=512,             # reduced from 1024
        n_epochs=10,                # reduced from 20, less overfitting per rollout
        gamma=0.99,                 # slightly higher, better long-term credit
        gae_lambda=0.95,            # SB3 default, more stable advantage estimates
        clip_range=0.2,             # explicit, SB3 default
        ent_coef=0.01,              # small entropy bonus helps exploration
        tensorboard_log="../runs/SB3_PPO_v2/",
        verbose=1,
        seed=seed
    )

checkpoint_callback = CheckpointCallback(
    save_freq=25000, 
    save_path='../models/SB3_PPO_v2/',
    name_prefix='rl_model'
)

# Set total_timesteps to exactly 4000,000
model.learn(total_timesteps=4000000, callback=[checkpoint_callback])

# Final save to ensure you have the exact end state
model.save(f"../models/SB3_PPO_v2/rl_model_4000000_steps")