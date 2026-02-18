from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import VecFrameStack

seed = 1

env_kwargs = dict(continuous=True)
env = make_vec_env("CarRacing-v3", env_kwargs=env_kwargs, n_envs=8, seed=seed)
env = VecFrameStack(env, n_stack=4) # stacking 4 frames so that the model can motion
model = PPO(policy="CnnPolicy",
            env=env,
            learning_rate=1e-3,
            n_steps=2048,
            batch_size=1024,
            n_epochs=20,
            gamma=0.98,
            gae_lambda=0.8,
            tensorboard_log="../runs/SB3_PPO_STACKED/",
            verbose=1,
            seed=seed)

checkpoint_callback = CheckpointCallback(save_freq=1e4, save_path='../models/SB3_PPO_STACKED/',)
model.learn(total_timesteps=1e6, callback=[checkpoint_callback])

steps = 0
while True:
    model.learn(total_timesteps=1e5)
    steps += 1e5
    model.save(f'../models/SB3_PPO_SEED_{seed}_STEPS_{steps:.0}.zip'.replace('+', ''))
