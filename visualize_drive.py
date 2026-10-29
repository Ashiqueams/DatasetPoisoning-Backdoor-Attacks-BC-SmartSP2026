import argparse
import importlib
import numpy as np
import torch
import yaml
import gymnasium as gym
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description="Watch a trained BC model drive, clean vs triggered.")
parser.add_argument('--run', required=True, help="run name key in config.yaml, e.g. run51_steer_clean_right")
parser.add_argument('--poison_level', type=int, required=True)
parser.add_argument('--model_seed', type=int, default=0)
parser.add_argument('--episode_seed', type=int, default=3_000_000)
parser.add_argument('--max_rollout_len', type=int, default=1000)
parser.add_argument('--mode', choices=['both', 'clean', 'triggered'], default='both',
                     help="clean = never show trigger, triggered = patch stamped on every frame "
                          "(matches the 'ALL_POISONED' test-time eval used for the acc_mean/control-rate numbers)")
parser.add_argument('--patch_size', type=int, default=3, help="trigger patch size in pixels (top-left corner)")
parser.add_argument('--patch_color', type=int, nargs=3, default=[255, 0, 0], metavar=('R', 'G', 'B'))
parser.add_argument('--delay', type=float, default=0.02, help="seconds paused per frame; raise to watch slower")
args = parser.parse_args()

with open("config.yaml") as f:
    cfg = yaml.safe_load(f)[args.run]

MODEL_DIR = cfg["model_dir"]
loss_type = cfg.get("loss_type", "mse")
module_name = "policyNetwork_nll" if loss_type == "nll" else "policyNetwork_mse"
PolicyNetwork = importlib.import_module(module_name).PolicyNetwork
print(f"[INFO] run={args.run} loss_type={loss_type} -> using {module_name}.PolicyNetwork")

device = torch.device(
    "mps" if torch.backends.mps.is_available()
    else ("cuda" if torch.cuda.is_available() else "cpu")
)

model = PolicyNetwork().to(device)
ckpt_path = f"{MODEL_DIR}/BC_P_{args.poison_level}_SEED_{args.model_seed}.pt"
model.load_state_dict(torch.load(ckpt_path, weights_only=True, map_location=device))
model.eval()
print(f"[INFO] loaded {ckpt_path}")


def make_env(ep_seed):
    env = gym.make('CarRacing-v3', continuous=True, domain_randomize=False)
    env.reset(seed=ep_seed)
    env.action_space.seed(ep_seed)
    env.observation_space.seed(ep_seed)
    return env


def add_trigger(image, patch_size, color):
    trojaned = image.copy()
    trojaned[:patch_size, :patch_size] = np.array(color, dtype=np.uint8)
    return trojaned


def classify_episode(terminated, truncated, last_reward):
    if terminated and last_reward <= -90:
        return "crashed_offtrack"
    if terminated:
        return "finished_early"
    return "timed_out"


def make_live_display(label):
    plt.ion()
    fig, ax = plt.subplots()
    im = ax.imshow(np.zeros((96, 96, 3), dtype=np.uint8))
    ax.axis('off')
    return fig, ax, im


def update_display(ax, im, obs, action, ts, label):
    im.set_data(obs)
    ax.set_title(f"{label} | t={ts} | steer={action[0]:.2f} gas={action[1]:.2f} brake={action[2]:.2f}")
    plt.pause(args.delay)


def run_episode(trigger, label):
    env = make_env(args.episode_seed)
    obs, _ = env.reset(seed=args.episode_seed)
    fig, ax, im = make_live_display(label)

    ts, cum_reward = 0, 0.0
    terminated, truncated, last_reward = False, False, 0.0

    while ts < args.max_rollout_len:
        display_obs = add_trigger(obs, args.patch_size, args.patch_color) if trigger else obs
        action, _ = model.predict([display_obs], device=device)
        update_display(ax, im, display_obs, action[0], ts, label)
        obs, reward, terminated, truncated, _ = env.step(action[0])
        cum_reward += reward
        last_reward = reward
        ts += 1
        if terminated:
            break

    tile_frac = env.unwrapped.tile_visited_count / len(env.unwrapped.track)
    env.close()
    plt.close(fig)

    outcome = classify_episode(terminated, truncated, last_reward)
    print(f"[{label}] reward={cum_reward:.1f}  outcome={outcome}  tile_frac={tile_frac:.3f}  steps={ts}")
    return cum_reward, outcome, tile_frac


if __name__ == "__main__":
    results = {}
    if args.mode in ('both', 'clean'):
        results['clean'] = run_episode(trigger=False, label="CLEAN (no trigger)")
    if args.mode in ('both', 'triggered'):
        results['triggered'] = run_episode(trigger=True, label="TRIGGERED (patch every frame)")

    if 'clean' in results and 'triggered' in results:
        dr = results['triggered'][0] - results['clean'][0]
        print(f"\n[SUMMARY] reward delta (triggered - clean) = {dr:+.1f}")
