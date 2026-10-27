import argparse
import time
import os
import torch
import numpy as np
import gymnasium as gym
import yaml
import matplotlib.pyplot as plt
from policyNetwork_nll import PolicyNetwork
from torch.distributions import Normal

parser = argparse.ArgumentParser()
parser.add_argument('--run', required=True)
parser.add_argument('--poison_level', type=int, required=True)
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--max_rollout_len', type=int, default=1000)
parser.add_argument('--num_rollouts', type=int, default=2)
parser.add_argument('--model_seeds', nargs='+', type=int, default=None)
parser.add_argument('--run_dir', default=None)
parser.add_argument('--sustain_len', type=int, default=25, help="length of one continuous burst")
parser.add_argument('--budget', type=int, default=50)
parser.add_argument('--warmup_steps', type=int, default=60, help="skip CarRacing's zoom-in intro")
parser.add_argument('--gate_direction', choices=['hard_left', 'hard_right'], default='hard_left',
                     help="hard_left: attack on left-turn-or-straight (matches run51-54 clean-label-right/attack-left). "
                          "hard_right: attack on right-turn-or-straight (matches run49 dirty-label direction).")
parser.add_argument('--steer_bound', type=float, default=-0.3,
                     help="for hard_left: fires when clean_steer <= steer_bound (left turn OR straight, excludes clear hard-right). "
                          "for hard_right: fires when clean_steer >= -steer_bound.")
parser.add_argument('--show_input', action='store_true')
args = parser.parse_args()

with open("config.yaml") as f:
    cfg = yaml.safe_load(f)[args.run]

MODEL_DIR = cfg["model_dir"]
MODEL_SEEDS = args.model_seeds if args.model_seeds is not None else cfg["model_seeds"]
POISON_LEVEL = args.poison_level
THRESHOLDS_PATH = f"entropy_thresholds_steer_{args.run}_P{POISON_LEVEL}.npy"

TTT_BASE_SEED = 3_000_000
torch.manual_seed(args.seed)
np.random.seed(args.seed)

device = torch.device(
    "mps" if torch.backends.mps.is_available()
    else ("cuda" if torch.cuda.is_available() else "cpu")
)

def make_env(ep_seed):
    env = gym.make('CarRacing-v3', continuous=True, domain_randomize=False)
    env.reset(seed=ep_seed)
    env.action_space.seed(ep_seed)
    env.observation_space.seed(ep_seed)
    return env

def add_trigger(image):
    trojaned = image.copy()
    trojaned[:3, :3] = np.array([255, 0, 0], dtype=np.uint8)
    return trojaned

def classify_episode(terminated, truncated, last_reward):
    if terminated and last_reward <= -90:
        return "crashed_offtrack"
    if terminated:
        return "finished_early"
    return "timed_out"

def gate_fires(clean_steer, steer_bound, gate_direction):
    if gate_direction == 'hard_left':
        return clean_steer <= steer_bound
    else:  # hard_right
        return clean_steer >= -steer_bound

def is_on_track(env):
    car = env.unwrapped.car
    return any(len(w.tiles) > 0 for w in car.wheels)

def make_live_display():
    plt.ion()
    fig, ax = plt.subplots()
    im = ax.imshow(np.zeros((96, 96, 3), dtype=np.uint8))
    ax.axis('off')
    ax.set_title("What the model sees")
    return im

def update_display(im, obs):
    im.set_data(obs)
    plt.pause(0.001)

def predict_with_entropy(model, obs, device):
    obs_array = np.array(obs)
    if obs_array.ndim == 3:
        obs_array = obs_array[np.newaxis, ...]
    obs_tensor = torch.from_numpy(obs_array).float().to(device) / 255.0
    obs_tensor = obs_tensor.permute(0, 3, 1, 2)
    with torch.no_grad():
        mean, log_var = model.forward(obs_tensor)
        std = torch.exp(0.5 * log_var)
        entropy = Normal(mean, std).entropy().sum(dim=-1)
    return mean.cpu().numpy(), entropy.cpu().numpy()

def run_clean_control(model, episode_seeds, max_rollout_len, device):
    rewards, outcomes, tile_fracs = [], [], []
    for attempt_seed in episode_seeds:
        env = make_env(attempt_seed)
        obs, _ = env.reset(seed=attempt_seed)
        ts, cum_reward = 0, 0.0
        terminated, truncated, last_reward = False, False, 0.0
        while ts < max_rollout_len:
            action, _ = model.predict([obs], device=device)
            obs, reward, terminated, truncated, _ = env.step(action[0])
            cum_reward += reward
            last_reward = reward
            ts += 1
            if terminated:
                break
        tile_frac = env.unwrapped.tile_visited_count / len(env.unwrapped.track)
        env.close()
        rewards.append(cum_reward)
        outcomes.append(classify_episode(terminated, truncated, last_reward))
        tile_fracs.append(tile_frac)
    return rewards, outcomes, tile_fracs

def run_steer_entropy_attack(model, entropy_threshold, steer_bound, gate_direction, sustain_len, budget,
                              warmup_steps, episode_seeds, max_rollout_len, device, show_input=False):
    rewards, attacked_counts, outcomes, tile_fracs = [], [], [], []
    live_im = make_live_display() if show_input else None

    for attempt_seed in episode_seeds:
        env = make_env(attempt_seed)
        obs, _ = env.reset(seed=attempt_seed)

        ts, cum_reward, count_attacked = 0, 0.0, 0
        terminated, truncated, last_reward = False, False, 0.0
        in_attack, attack_left = False, 0

        while ts < max_rollout_len:
            clean_action, entropy = predict_with_entropy(model, [obs], device)
            clean_steer = clean_action[0][0]

            if (not in_attack) and (count_attacked < budget) and (ts >= warmup_steps) \
                    and gate_fires(clean_steer, steer_bound, gate_direction) \
                    and (entropy[0] <= entropy_threshold +1e-3) and is_on_track(env):
                in_attack = True
                attack_left = min(sustain_len, budget - count_attacked)
                print(f"  [t={ts}] ATTACK FIRE (clean_steer={clean_steer:.2f}, entropy={entropy[0]:.3f}, "
                      f"attacked_so_far={count_attacked})")

            if in_attack and attack_left > 0:
                if not is_on_track(env):
                    print(f"  [t={ts}] off-track mid-burst, ending attack early (attacked_so_far={count_attacked})")
                    in_attack = False
                    attack_left = 0
                    action = clean_action
                else:
                    triggered_obs = add_trigger(obs)
                    if live_im is not None:
                        update_display(live_im, triggered_obs)
                    action, _ = model.predict([triggered_obs], device=device)
                    count_attacked += 1
                    attack_left -= 1
                    if attack_left == 0:
                        in_attack = False
            else:
                if live_im is not None:
                    update_display(live_im, obs)
                action = clean_action

            obs, reward, terminated, truncated, _ = env.step(action[0])
            cum_reward += reward
            last_reward = reward
            ts += 1
            if terminated:
                break
        tile_frac = env.unwrapped.tile_visited_count / len(env.unwrapped.track)
        env.close()

        rewards.append(cum_reward)
        attacked_counts.append(count_attacked)
        outcomes.append(classify_episode(terminated, truncated, last_reward))
        tile_fracs.append(tile_frac)

    return rewards, attacked_counts, outcomes, tile_fracs

def save_results(save_dir, name, rewards, attacked_counts=None, outcomes=None, tile_fracs=None):
    os.makedirs(save_dir, exist_ok=True)
    np.save(f"{save_dir}/{name}_rewards.npy", np.array(rewards))
    if attacked_counts is not None:
        np.save(f"{save_dir}/{name}_attacked_counts.npy", np.array(attacked_counts))
    if outcomes is not None:
        np.save(f"{save_dir}/{name}_outcomes.npy", np.array(outcomes))
    if tile_fracs is not None:
        np.save(f"{save_dir}/{name}_tile_fracs.npy", np.array(tile_fracs))
    print(f"[INFO] Saved {name} -> {save_dir}/")

if __name__ == "__main__":
    if args.run_dir is not None:
        run_dir = args.run_dir
    else:
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        run_dir = f"./ttt_steer_entropy_runs/{args.run}/P{POISON_LEVEL}_seed{args.seed}_time{timestamp}"

    all_thresholds = np.load(THRESHOLDS_PATH, allow_pickle=True).item()
    base_seed = TTT_BASE_SEED + args.seed
    episode_seeds = [base_seed + i for i in range(args.num_rollouts)]

    for model_seed in MODEL_SEEDS:
        print(f"\n===== Model seed {model_seed} =====")
        model = PolicyNetwork().to(device)
        model.load_state_dict(torch.load(
            f"{MODEL_DIR}/BC_P_{POISON_LEVEL}_SEED_{model_seed}.pt", weights_only=True, map_location=device
        ))
        model.eval()

        seed_dir = f"{run_dir}/model_seed{model_seed}"
        # entropy_threshold = all_thresholds[model_seed][args.budget]
        entropy_threshold = 0
        clean_rewards, clean_outcomes, clean_tiles = run_clean_control(
            model, episode_seeds, args.max_rollout_len, device
        )
        save_results(seed_dir, "clean_control", clean_rewards, outcomes=clean_outcomes, tile_fracs=clean_tiles)
        clean_crash_rate = np.mean([o == "crashed_offtrack" for o in clean_outcomes])
        print(f"Clean control: mean={np.mean(clean_rewards):.1f}, crash_rate={clean_crash_rate*100:.1f}%, "
              f"mean_tile_frac={np.mean(clean_tiles):.3f}")
        print(f"  per-episode rewards: {[round(r, 1) for r in clean_rewards]}")

        rewards, attacked_counts, outcomes, tiles = run_steer_entropy_attack(
            model, entropy_threshold, args.steer_bound, args.gate_direction, args.sustain_len, args.budget,
            args.warmup_steps, episode_seeds, args.max_rollout_len, device, show_input=args.show_input
        )
        attack_caused = sum(1 for cl, at in zip(clean_outcomes, outcomes)
                             if cl != "crashed_offtrack" and at == "crashed_offtrack")
        crash_rate = np.mean([o == "crashed_offtrack" for o in outcomes])
        save_results(seed_dir, "steer_entropy_attack", rewards, attacked_counts, outcomes, tiles)
        print(f"Steer+entropy attack (gate={args.gate_direction}, steer_bound={args.steer_bound}, "
              f"entropy_threshold={entropy_threshold:.3f}, budget={args.budget}, sustain={args.sustain_len}, "
              f"warmup={args.warmup_steps}): mean={np.mean(rewards):.1f} ± {np.std(rewards):.1f}, "
              f"crash_rate={crash_rate*100:.1f}%, attack_caused_crash_rate={attack_caused/len(episode_seeds)*100:.1f}%, "
              f"avg_attacked={np.mean(attacked_counts):.1f}, mean_tile_frac={np.mean(tiles):.3f}")
        print(f"  per-episode rewards: {[round(r, 1) for r in rewards]}")

        if args.show_input:
            plt.close('all')
