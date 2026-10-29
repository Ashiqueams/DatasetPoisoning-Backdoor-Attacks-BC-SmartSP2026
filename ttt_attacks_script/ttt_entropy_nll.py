# import argparse
# import time
# import os
# import torch
# import numpy as np
# import gymnasium as gym
# import yaml
# from policyNetwork_nll import PolicyNetwork
# from torch.distributions import Normal

# parser = argparse.ArgumentParser()
# parser.add_argument('--run', required=True, help="run name from config.yaml")
# parser.add_argument('--poison_level', type=int, required=True)
# parser.add_argument('--seed', type=int, default=0)
# parser.add_argument('--max_rollout_len', type=int, default=1000)
# parser.add_argument('--num_rollouts', type=int, default=80)
# parser.add_argument('--model_seeds', nargs='+', type=int, default=None)
# parser.add_argument('--save_frames', type=int, default=0, help="if >0, save up to this many triggered frames for inspection")
# parser.add_argument('--run_dir', default=None)
# parser.add_argument('--save_frame_gap', type=int, default=50, help="minimum timesteps between saved frames, to avoid near-duplicate saves")
# parser.add_argument('--max_frames_per_rollout', type=int, default=2, help="cap on saved frames from any single episode, to force variety across tracks")
# args = parser.parse_args()

# with open("config.yaml") as f:
#     cfg = yaml.safe_load(f)[args.run]

# MODEL_DIR = cfg["model_dir"]
# MODEL_SEEDS = args.model_seeds if args.model_seeds is not None else cfg["model_seeds"]
# POISON_LEVEL = args.poison_level
# THRESHOLDS_PATH = f"entropy_thresholds_{args.run}_P{POISON_LEVEL}.npy"
# BUDGETS = [5, 10, 15, 50, 100, 250, 500]

# TTT_BASE_SEED = 3_000_000   # SAME base as random-timing -- gives both strategies identical tracks

# torch.manual_seed(args.seed)
# np.random.seed(args.seed)

# device = torch.device(
#     "mps" if torch.backends.mps.is_available()
#     else ("cuda" if torch.cuda.is_available() else "cpu")
# )

# def is_target_action(actions):
#     actions = np.asarray(actions)
#     if actions.ndim == 1:
#         actions = actions[None, :]
#     gas = actions[:, 1]
#     brake = actions[:, 2]
#     return (gas >= 0.5) & (brake < 0.1)

# def make_env(ep_seed):
#     env = gym.make('CarRacing-v3', continuous=True, domain_randomize=False)
#     env.reset(seed=ep_seed)
#     env.action_space.seed(ep_seed)
#     env.observation_space.seed(ep_seed)
#     return env

# def add_trigger(image):
#     trojaned = image.copy()
#     trojaned[:3, :3] = np.array([255, 0, 0], dtype=np.uint8)
#     return trojaned

# def predict_with_entropy(model, obs, device):
#     obs_array = np.array(obs)
#     if obs_array.ndim == 3:
#         obs_array = obs_array[np.newaxis, ...]
#     obs_tensor = torch.from_numpy(obs_array).float().to(device) / 255.0
#     obs_tensor = obs_tensor.permute(0, 3, 1, 2)
#     with torch.no_grad():
#         mean, log_var = model.forward(obs_tensor)
#         std = torch.exp(0.5 * log_var)
#         entropy = Normal(mean, std).entropy().sum(dim=-1)
#     return mean.cpu().numpy(), entropy.cpu().numpy()

# def run_clean_control(model, episode_seeds, max_rollout_len, device):
#     rewards, outcomes = [], []
#     for attempt_seed in episode_seeds:
#         env = make_env(attempt_seed)
#         obs, _ = env.reset(seed=attempt_seed)
#         ts, cum_reward = 0, 0.0
#         terminated, truncated, last_reward = False, False, 0.0
#         while ts < max_rollout_len:
#             mean, _ = predict_with_entropy(model, [obs], device)
#             obs, reward, terminated, truncated, _ = env.step(mean[0])
#             cum_reward += reward
#             last_reward = reward
#             ts += 1
#             if terminated:
#                 break
#         env.close()
#         rewards.append(cum_reward)
#         outcomes.append(classify_episode(terminated, truncated, last_reward))
#     return rewards, outcomes

# def classify_episode(terminated, truncated, last_reward):
#     if terminated and last_reward <= -90:
#         return "crashed_offtrack"      # CarRacing gives -100 on out-of-bounds termination
#     if terminated:
#         return "finished_early"        # all tiles visited — a good outcome, not a crash
#     return "timed_out"                 # truncated at max_rollout_len, the normal case

# def run_entropy_budget_attack(model, budget, threshold, episode_seeds, max_rollout_len, device,
#                                frame_save_dir=None, max_frames_to_save=0, save_frame_gap=50, max_frames_per_rollout=2):
#     rewards, attacked_counts, outcomes = [], [], []
#     frames_saved = 0

#     for attempt_seed in episode_seeds:
#         env = make_env(attempt_seed)
#         obs, _ = env.reset(seed=attempt_seed)

#         ts, cum_reward, count_attacked = 0, 0.0, 0
#         terminated, truncated, last_reward = False, False, 0.0
#         last_saved_ts = -10**9
#         frames_saved_this_rollout = 0

#         while ts < max_rollout_len:
#             clean_action, entropy = predict_with_entropy(model, [obs], device)
#             is_non_target = not is_target_action(clean_action[0])[0]
#             should_attack = (entropy[0] <= threshold) and is_non_target and (count_attacked < budget)

#             if should_attack:
#                 if (frame_save_dir is not None and frames_saved < max_frames_to_save
#                         and (ts - last_saved_ts) >= save_frame_gap
#                         and frames_saved_this_rollout < max_frames_per_rollout):
#                     os.makedirs(frame_save_dir, exist_ok=True)
#                     np.save(f"{frame_save_dir}/frame_{frames_saved:03d}.npy", obs)
#                     frames_saved += 1
#                     frames_saved_this_rollout += 1
#                     last_saved_ts = ts
#                 triggered_obs = add_trigger(obs)
#                 action, _ = model.predict([triggered_obs], device=device)
#                 count_attacked += 1
#             else:
#                 action = clean_action

#             obs, reward, terminated, truncated, _ = env.step(action[0])
#             cum_reward += reward
#             last_reward = reward
#             ts += 1
#             if terminated:
#                 break
#         env.close()

#         rewards.append(cum_reward)
#         attacked_counts.append(count_attacked)
#         outcomes.append(classify_episode(terminated, truncated, last_reward))

#     return rewards, attacked_counts, outcomes

# def save_results(save_dir, name, rewards, attacked_counts=None, outcomes=None):
#     os.makedirs(save_dir, exist_ok=True)
#     np.save(f"{save_dir}/{name}_rewards.npy", np.array(rewards))
#     if attacked_counts is not None:
#         np.save(f"{save_dir}/{name}_attacked_counts.npy", np.array(attacked_counts))
#     if outcomes is not None:
#         np.save(f"{save_dir}/{name}_outcomes.npy", np.array(outcomes))
#     print(f"[INFO] Saved {name} -> {save_dir}/")


# if __name__ == "__main__":
#     if args.run_dir is not None:
#         run_dir = args.run_dir
#     else:
#         timestamp = time.strftime("%Y%m%d-%H%M%S")
#         run_dir = f"./ttt_entropy_runs/{args.run}/P{POISON_LEVEL}_seed{args.seed}_time{timestamp}"

#     all_thresholds = np.load(THRESHOLDS_PATH, allow_pickle=True).item()
#     base_seed = TTT_BASE_SEED + args.seed
#     episode_seeds = [base_seed + i for i in range(args.num_rollouts)]

#     for model_seed in MODEL_SEEDS:
#         print(f"\n===== Model seed {model_seed} =====")
#         model = PolicyNetwork().to(device)
#         model.load_state_dict(torch.load(
#             f"{MODEL_DIR}/BC_P_{POISON_LEVEL}_SEED_{model_seed}.pt", weights_only=True, map_location=device
#         ))
#         model.eval()

#         seed_dir = f"{run_dir}/model_seed{model_seed}"
#         seed_thresholds = all_thresholds[model_seed]

#         clean_rewards, clean_outcomes = run_clean_control(model, episode_seeds, args.max_rollout_len, device)
#         save_results(seed_dir, "clean_control", clean_rewards, outcomes=clean_outcomes)
#         clean_crash_rate = np.mean([o == "crashed_offtrack" for o in clean_outcomes])
#         print(f"Clean control: mean={np.mean(clean_rewards):.1f}, crash_rate={clean_crash_rate*100:.1f}%")

#         for budget in BUDGETS:
#             threshold = seed_thresholds[budget]
#             rewards, attacked_counts, outcomes = run_entropy_budget_attack(
#                 model, budget, threshold, episode_seeds, args.max_rollout_len, device,
#                 frame_save_dir=f"{seed_dir}/triggered_frames_budget{budget}" if args.save_frames > 0 else None,
#                 max_frames_to_save=args.save_frames, save_frame_gap=args.save_frame_gap,
#                 max_frames_per_rollout=args.max_frames_per_rollout
#             )
#             attack_caused = sum(1 for cl, at in zip(clean_outcomes, outcomes)
#                                  if cl != "crashed_offtrack" and at == "crashed_offtrack")
#             crash_rate = np.mean([o == "crashed_offtrack" for o in outcomes])
#             save_results(seed_dir, f"entropy_budget{budget}", rewards, attacked_counts, outcomes)
#             print(f"Entropy, budget={budget}: mean={np.mean(rewards):.1f} ± {np.std(rewards):.1f}, "
#                   f"crash_rate={crash_rate*100:.1f}%, attack_caused_crash_rate={attack_caused/len(episode_seeds)*100:.1f}%, "
#                   f"avg_attacked={np.mean(attacked_counts):.1f}")
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
parser.add_argument('--run', required=True, help="run name from config.yaml")
parser.add_argument('--poison_level', type=int, required=True)
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--max_rollout_len', type=int, default=1000)
parser.add_argument('--num_rollouts', type=int, default=80)
parser.add_argument('--model_seeds', nargs='+', type=int, default=None)
parser.add_argument('--save_frames', type=int, default=0, help="if >0, save up to this many triggered frames for inspection")
parser.add_argument('--run_dir', default=None)
parser.add_argument('--save_frame_gap', type=int, default=50, help="minimum timesteps between saved frames, to avoid near-duplicate saves")
parser.add_argument('--max_frames_per_rollout', type=int, default=2, help="cap on saved frames from any single episode, to force variety across tracks")
parser.add_argument('--show_input', action='store_true', help="live window showing exactly what the model sees, trigger included")
args = parser.parse_args()

with open("config.yaml") as f:
    cfg = yaml.safe_load(f)[args.run]

MODEL_DIR = cfg["model_dir"]
MODEL_SEEDS = args.model_seeds if args.model_seeds is not None else cfg["model_seeds"]
POISON_LEVEL = args.poison_level
THRESHOLDS_PATH = f"entropy_thresholds_{args.run}_P{POISON_LEVEL}.npy"
BUDGETS = [5, 10, 15, 50, 100, 250, 500, 1000]
# BUDGETS = [ 100]
TTT_BASE_SEED = 3_000_000   # SAME base as random-timing -- gives both strategies identical tracks

torch.manual_seed(args.seed)
np.random.seed(args.seed)

device = torch.device(
    "mps" if torch.backends.mps.is_available()
    else ("cuda" if torch.cuda.is_available() else "cpu")
)

def is_target_action(actions):
    actions = np.asarray(actions)
    if actions.ndim == 1:
        actions = actions[None, :]
    gas = actions[:, 1]
    brake = actions[:, 2]
    return (gas >= 0.5) & (brake < 0.1)

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
    rewards, outcomes = [], []
    for attempt_seed in episode_seeds:
        env = make_env(attempt_seed)
        obs, _ = env.reset(seed=attempt_seed)
        ts, cum_reward = 0, 0.0
        terminated, truncated, last_reward = False, False, 0.0
        while ts < max_rollout_len:
            mean, _ = predict_with_entropy(model, [obs], device)
            obs, reward, terminated, truncated, _ = env.step(mean[0])
            cum_reward += reward
            last_reward = reward
            ts += 1
            if terminated:
                break
        env.close()
        rewards.append(cum_reward)
        outcomes.append(classify_episode(terminated, truncated, last_reward))
    return rewards, outcomes

def classify_episode(terminated, truncated, last_reward):
    if terminated and last_reward <= -90:
        return "crashed_offtrack"      # CarRacing gives -100 on out-of-bounds termination
    if terminated:
        return "finished_early"        # all tiles visited — a good outcome, not a crash
    return "timed_out"                 # truncated at max_rollout_len, the normal case

def run_entropy_budget_attack(model, budget, threshold, episode_seeds, max_rollout_len, device,
                               frame_save_dir=None, max_frames_to_save=0, save_frame_gap=50, max_frames_per_rollout=2,
                               show_input=False):
    rewards, attacked_counts, outcomes = [], [], []
    frames_saved = 0
    live_im = make_live_display() if show_input else None

    for attempt_seed in episode_seeds:
        env = make_env(attempt_seed)
        obs, _ = env.reset(seed=attempt_seed)

        ts, cum_reward, count_attacked = 0, 0.0, 0
        terminated, truncated, last_reward = False, False, 0.0
        last_saved_ts = -10**9
        frames_saved_this_rollout = 0

        while ts < max_rollout_len:
            clean_action, entropy = predict_with_entropy(model, [obs], device)
            is_non_target = not is_target_action(clean_action[0])[0]
            should_attack = (entropy[0] <= threshold) and is_non_target and (count_attacked < budget)

            if should_attack:
                print(f"  [t={ts}] ATTACK FIRE (entropy={entropy[0]:.3f} <= {threshold:.3f})")
                if (frame_save_dir is not None and frames_saved < max_frames_to_save
                        and (ts - last_saved_ts) >= save_frame_gap
                        and frames_saved_this_rollout < max_frames_per_rollout):
                    os.makedirs(frame_save_dir, exist_ok=True)
                    np.save(f"{frame_save_dir}/frame_{frames_saved:03d}.npy", obs)
                    frames_saved += 1
                    frames_saved_this_rollout += 1
                    last_saved_ts = ts
                triggered_obs = add_trigger(obs)
                if live_im is not None:
                    update_display(live_im, triggered_obs)
                action, _ = model.predict([triggered_obs], device=device)
                count_attacked += 1
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
        env.close()

        rewards.append(cum_reward)
        attacked_counts.append(count_attacked)
        outcomes.append(classify_episode(terminated, truncated, last_reward))

    return rewards, attacked_counts, outcomes

def save_results(save_dir, name, rewards, attacked_counts=None, outcomes=None):
    os.makedirs(save_dir, exist_ok=True)
    np.save(f"{save_dir}/{name}_rewards.npy", np.array(rewards))
    if attacked_counts is not None:
        np.save(f"{save_dir}/{name}_attacked_counts.npy", np.array(attacked_counts))
    if outcomes is not None:
        np.save(f"{save_dir}/{name}_outcomes.npy", np.array(outcomes))
    print(f"[INFO] Saved {name} -> {save_dir}/")


if __name__ == "__main__":
    if args.run_dir is not None:
        run_dir = args.run_dir
    else:
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        run_dir = f"./ttt_entropy_runs/{args.run}/P{POISON_LEVEL}_seed{args.seed}_time{timestamp}"

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
        seed_thresholds = all_thresholds[model_seed]

        clean_rewards, clean_outcomes = run_clean_control(model, episode_seeds, args.max_rollout_len, device)
        save_results(seed_dir, "clean_control", clean_rewards, outcomes=clean_outcomes)
        clean_crash_rate = np.mean([o == "crashed_offtrack" for o in clean_outcomes])
        print(f"Clean control: mean={np.mean(clean_rewards):.1f}, crash_rate={clean_crash_rate*100:.1f}%")
        print(f"  per-episode rewards: {[round(r, 1) for r in clean_rewards]}")

        for budget in BUDGETS:
            threshold = seed_thresholds[budget]
            rewards, attacked_counts, outcomes = run_entropy_budget_attack(
                model, budget, threshold, episode_seeds, args.max_rollout_len, device,
                frame_save_dir=f"{seed_dir}/triggered_frames_budget{budget}" if args.save_frames > 0 else None,
                max_frames_to_save=args.save_frames, save_frame_gap=args.save_frame_gap,
                max_frames_per_rollout=args.max_frames_per_rollout,
                show_input=args.show_input
            )
            attack_caused = sum(1 for cl, at in zip(clean_outcomes, outcomes)
                                 if cl != "crashed_offtrack" and at == "crashed_offtrack")
            crash_rate = np.mean([o == "crashed_offtrack" for o in outcomes])
            save_results(seed_dir, f"entropy_budget{budget}", rewards, attacked_counts, outcomes)
            print(f"Entropy, budget={budget}: mean={np.mean(rewards):.1f} ± {np.std(rewards):.1f}, "
                  f"crash_rate={crash_rate*100:.1f}%, attack_caused_crash_rate={attack_caused/len(episode_seeds)*100:.1f}%, "
                  f"avg_attacked={np.mean(attacked_counts):.1f}")
            print(f"  per-episode rewards: {[round(r, 1) for r in rewards]}")
            if args.show_input:
                plt.close('all')
