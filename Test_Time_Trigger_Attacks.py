__author__ = "akansha_kalra"
import os
from copy import deepcopy
from collections import defaultdict
import torch
import numpy as np
import random
import gymnasium as gym
from stable_baselines3.common.utils import set_random_seed
from policynetwork import PolicyNetwork
# from torch.distributions import Categorical
import argparse
import time
import torch.nn.functional as F
from torch.distributions import Normal

parser = argparse.ArgumentParser()

parser.add_argument('--seed', default=0, help="Seed for experiments")
parser.add_argument('--max_rollout_len', default=1000, help="max length of each rolloutfor test time attack")
parser.add_argument('--num_rollouts',default=100,help="number of rollouts to run test time attacks")
parser.add_argument('--attack_budget',default=100,help="budget,aka no of test time attacks allowed in a single rollout b")
parser.add_argument('--entropy_threshold', type=float, default=0.5, help="entropy threshold for entropy-based attacks")
parser.add_argument('--attack_types_ls', nargs='+',default=["red_random","red_entropy","gaussian_random","gaussian_entropy","unattacked_poisoned_red","unattacked_poison_gaussian"],help="which attacks and unattacked evals on BC models trained on poisoned datasets to run")
parser.add_argument('--base_save_model_dir', default="TTT_attacks/", help="where to save trained model")
parser.add_argument('--red_BC_policy_path', default="../models/BC_red1_cameraready_run2/",
                    help="path where BC policy trained on data poisoned with red patch is stored")
parser.add_argument('--gaussian_BC_policy_path', default="saved_models/",
                    help="path where BC policy trained on data poisoned with gaussian patch is stored")


args = parser.parse_args()
seed = int(args.seed)
max_rollout_len=int(args.max_rollout_len)
num_onpolicy_rollouts=int(args.num_rollouts)
attack_budget=int(args.attack_budget)
entropy_threshold = args.entropy_threshold
attack_types_ls=args.attack_types_ls
base_save_model_dir = args.base_save_model_dir
red_BC_policy_path = args.red_BC_policy_path
gaussian_BC_policy_path = args.gaussian_BC_policy_path

torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
set_random_seed(seed)
print(f'Setting seed to {seed} for Pytorch,NP,SB3,random')

def make_env(seed):
    env = gym.make('CarRacing-v3', continuous=True, domain_randomize=False)
    env.reset(seed=seed)
    env.action_space.seed(seed)
    env.observation_space.seed(seed)
    return env

# def entropy(action_probs):
#     dist = Categorical(probs=action_probs)
#     entropy = dist.entropy().item()
#     return entropy

def is_target_action_continuous(a, steer_eps = 0.2, gas_min=0.8, brake_max = 0.1):
    a = np.asarray(a, dtype=np.float32).reshape(-1)
    steer, gas, brake = float(a[0]), float(a[1]), float(a[2])
    return (gas > gas_min) and (brake < brake_max) and (abs(steer) < steer_eps)

def gaussian_action_and_entropy(policy, obs_tensor):
    #obs_tensor torch float tensor shape (1,3,96,96) on device
    action_tensor, entropy_tensor = policy.sample_action_and_entropy(obs_tensor)
    action = action_tensor[0].detach().cpu().numpy().astype(np.float32)
    ent = float(entropy_tensor[0].detach().cpu().item())
    
    return action, ent
    
def add_adv_patch(obs,attack_strength=3):
    trojaned=deepcopy(obs)
    trojaned[:int(attack_strength), :int(attack_strength)] = np.array([255, 0, 0])
    return trojaned

def final_correct_seeded_red_random_attack_with_budget_fixed_len_changed_random_sampling(backdoor_policy,
        num_onpolicy_rollouts=5,budget=300,fixed_rollout_len=1000):

    print('----' * 15)
    print(f"Starting random budget attacks")
    device = torch.device(
        "mps" if torch.backends.mps.is_available()
        else ("cuda" if torch.cuda.is_available() else "cpu"))
    backdoor_policy.to(device)
    backdoor_policy.eval()
    count_total_states_dict = defaultdict(list)
    count_attacked_state_dict = defaultdict(list)
    attacked_state_idx_dict = defaultdict(list)
    reward_dict = defaultdict(list)
    episode_seed_dict = defaultdict(int)

    accepted_count = 0   # number of accepted (kept) rollouts
    attempt_idx = 0      # how many rollout attempts we've made (used to derive seed for each attempt)

    while accepted_count < num_onpolicy_rollouts:
        print(f"----------Attempt {attempt_idx} (accepted {accepted_count})------------------")
        ts = 0
        count_total_states = 0
        count_attacked_states = 0
        curr_attacked_state_idxs = []
        attempt_seed = seed + attempt_idx
        env = make_env(attempt_seed)
        obs, info = env.reset(seed=attempt_seed)  # attempt-specific seed
        cum_reward = 0
        random_sampled_attack_idxs = np.random.choice(a=fixed_rollout_len, size=budget, replace=False)
        random_sampled_attack_idxs=np.sort(random_sampled_attack_idxs).tolist()
        print(f'To be attacked states are {len(random_sampled_attack_idxs)} with idxs :{random_sampled_attack_idxs}')
        while ts < fixed_rollout_len:
            unnormalized_obs = obs
            if ts in random_sampled_attack_idxs:
                '''add attack'''
                attacked_obs = add_adv_patch(unnormalized_obs)
                attacked_obs = attacked_obs/255.0
                count_attacked_states += 1
                if attacked_obs.ndim == 3:
                    attacked_obs = np.transpose(attacked_obs, (2, 0, 1))  # HWC → CHW
                    attacked_obs = torch.tensor(attacked_obs, dtype=torch.float32).unsqueeze(0).to(device)
                # curr_attacked_action = backdoor_policy.forward(attacked_obs).argmax(dim=1)[0].cpu().numpy()
                action = backdoor_policy.forward(attacked_obs)[0].detach().cpu().numpy().astype(np.float32)
                obs, reward, terminated,truncated, _ = env.step(action)
                curr_attacked_state_idxs.append(ts)
            else:
                obs_nt = np.transpose((unnormalized_obs/255.0), (2,0,1))
                obs_t = torch.tensor(obs_nt, dtype=torch.float32).unsqueeze(0).to(device)
                # curr_action_prob_dist = backdoor_policy.forward(obs)
                # curr_pred_action = curr_action_prob_dist.argmax(dim=1)[0].cpu().numpy()
                action = backdoor_policy.forward(obs_t)
                action = action[0].detach().cpu().numpy().astype(np.float32)
                obs, reward, terminated,truncated, _ = env.step(action)

            cum_reward += reward
            ts += 1
            count_total_states += 1

        if (count_total_states == fixed_rollout_len) and (truncated is True) and (count_attacked_states == budget):

            print(f'Finished attacking {count_total_states} states reached episode len{fixed_rollout_len} and Truncated{truncated}')
            ep_idx = accepted_count  # index in the kept-rollouts dicts (0..num_onpolicy_rollouts-1)
            count_total_states_dict[ep_idx] = count_total_states
            count_attacked_state_dict[ep_idx] = count_attacked_states
            attacked_state_idx_dict[ep_idx] = curr_attacked_state_idxs
            reward_dict[ep_idx] = cum_reward
            episode_seed_dict[ep_idx] = attempt_seed

            print(
                f"Accepted rollout {ep_idx} (seed {attempt_seed}): Reward {cum_reward}, "
                f"attacked {count_attacked_states}/{count_total_states} states.")
            accepted_count += 1
        else:
            if terminated:
                print(
                    f'Terminated early for attempt idx{attempt_idx} with attempt seed {attempt_seed} reached episode len{ts}')
        attempt_idx += 1
        env.close()

    print('--!!--' * 10)
    print(f"RANDOM Reward dict {reward_dict} with length of rollouts {count_total_states_dict} "
          f"and states attacked {count_attacked_state_dict}")
    return reward_dict, count_total_states_dict, count_attacked_state_dict,attacked_state_idx_dict,episode_seed_dict

def final_correct_seeded_red_entropy_attack_with_budget_all_non_gas_ensure_fixed_len(backdoor_policy,entropy_threshold,num_onpolicy_rollouts=5,budget=300,fixed_rollout_len=1000):
    device = torch.device(
        "mps" if torch.backends.mps.is_available()
        else ("cuda" if torch.cuda.is_available() else "cpu"))
    backdoor_policy.to(device)
    backdoor_policy.eval()
    
    count_total_states_dict = defaultdict(list)
    count_attacked_state_dict = defaultdict(list)
    attacked_state_idx_dict = defaultdict(list)
    reward_dict = defaultdict(list)
    episode_seed_dict = defaultdict(int)
    
    accepted_count = 0
    attempt_idx=0
    
    while accepted_count < num_onpolicy_rollouts:

    # for episode_idx in range(num_onpolicy_rollouts):
    #     print(f"---------------------------Episode {episode_idx}------------------")
        ts = 0
        count_total_states = 0
        count_attacked_states = 0
        curr_attacked_state_idxs = []
        attempt_seed = seed + attempt_idx
        print(f"----------Attempting {attempt_idx} with seed {attempt_seed} and total accepted full len rollouts {accepted_count}------------------")
        env = make_env(attempt_seed)
        obs, info = env.reset(seed=attempt_seed)
        cum_reward = 0.0
        while ts < fixed_rollout_len:
            unnormalized_obs = obs
            obs_nt = np.transpose((unnormalized_obs/255.0),(2,0,1))
            obs_t = torch.tensor(obs_nt, dtype=torch.float32).unsqueeze(0).to(device)
            clean_action, ent = gaussian_action_and_entropy(backdoor_policy, obs_t)
            clean_is_target = is_target_action_continuous(clean_action)
            
            do_attack = (ent < entropy_threshold) and (count_attacked_states < budget) and (not clean_is_target)
            # curr_action_prob_dist = backdoor_policy.forward(obs)
            # curr_pred_action = curr_action_prob_dist.argmax(dim=1)[0].cpu().numpy()
            # curr_entropy = entropy(curr_action_prob_dist)

            #     '''gas should not be optimal/near optimal'''
            # if curr_entropy < entropy_threshold and count_attacked_states<budget and curr_pred_action!=3:
            if do_attack:
                '''add attack'''
                attacked_obs = add_adv_patch(unnormalized_obs)
                attacked_obs = attacked_obs/255.0
                # print(f"--------Attacking state at ts {ts} and current attacked state count {count_attacked_states}----------------")
                count_attacked_states += 1
                if attacked_obs.ndim == 3:
                    attacked_obs = np.transpose(attacked_obs, (2, 0, 1))  # HWC → CHW
                    attacked_obs = torch.tensor(attacked_obs, dtype=torch.float32).unsqueeze(0).to(device)
                attacked_action, _ = gaussian_action_and_entropy(backdoor_policy, attacked_obs)
                # action = backdoor_policy.forward(obs)
                # action = action[0].detach().cpu().numpy().astype(np.float32)
                obs, reward, terminated,truncated, _ = env.step(attacked_action)
                curr_attacked_state_idxs.append(ts)
                # print(f"True pred action {curr_pred_action} but taking test time trigger action {curr_attacked_action} at ts {ts} has reward {reward}")
            else:
                obs, reward, terminated, truncated, info = env.step(clean_action)
            # else:
            #     obs, reward, terminated, truncated, _ = env.step(curr_pred_action)
            #     # print(f"True pred action {curr_pred_action} at ts {ts} has reward {reward}")

            cum_reward += float(reward)
            ts += 1
            count_total_states += 1

        # if (count_total_states == fixed_rollout_len) and (truncated is True) and (count_attacked_states == budget):
        # As i have changed the rollout length:
        # if (count_total_states == fixed_rollout_len) and (count_attacked_states == budget):
        if (count_total_states >= 1):

            # if ts == fixed_rollout_len and truncated==True:
            ep_idx = accepted_count  # index in the kept-rollouts dicts (0..num_onpolicy_rollouts-1)
            count_total_states_dict[ep_idx] = count_total_states
            count_attacked_state_dict[ep_idx] = count_attacked_states
            attacked_state_idx_dict[ep_idx] = curr_attacked_state_idxs
            reward_dict[ep_idx] = cum_reward
            episode_seed_dict[ep_idx] = attempt_seed

            print(
                f"Accepted rollout {ep_idx} (seed {attempt_seed}): Reward {cum_reward}, "
                f"attacked {count_attacked_states}/{count_total_states} states.")
            accepted_count += 1
        else:
            if terminated:
                print(f'Terminated early for attempt idx{attempt_idx} with attempt seed {attempt_seed} reached episode len{ts}')
        attempt_idx += 1
        env.close()

    print('--!!--' * 10)
    print(
        f"ENTROPY Reward dict {reward_dict} with length of rollouts {count_total_states_dict} and states attacked {count_attacked_state_dict}")
    return reward_dict, count_total_states_dict, count_attacked_state_dict, attacked_state_idx_dict, episode_seed_dict



def add_adv_gaussian_patch_Deterministic(obs, attack_strength=3):
    patch_seed = 0
    rng_patch = np.random.default_rng(patch_seed)
    trojaned=deepcopy(obs)
    patch_size = 3
    fixed_gaussian_patch = np.clip(
        rng_patch.normal(loc=127, scale=30, size=(patch_size, patch_size, 3)), 0, 255
    ).astype(np.uint8)
    trojaned[:int(attack_strength), :int(attack_strength)] = fixed_gaussian_patch
    return trojaned

def final_correct_seeded_gaussian_random_attack_with_budget_fixed_len_changed_random_sampling(backdoor_policy, num_onpolicy_rollouts,budget=300,fixed_rollout_len=1000):
        print('----' * 15)
        print(f"Starting Gaussian random budget attacks")
        device = torch.device(
            "mps" if torch.backends.mps.is_available()
            else ("cuda" if torch.cuda.is_available() else "cpu"))
        backdoor_policy.to(device)
        backdoor_policy.eval()
        count_total_states_dict = defaultdict(list)
        count_attacked_state_dict = defaultdict(list)
        attacked_state_idx_dict = defaultdict(list)
        reward_dict = defaultdict(list)
        episode_seed_dict = defaultdict(int)
        accepted_count = 0  # number of accepted (kept) rollouts
        attempt_idx = 0  # how many rollout attempts we've made (used to derive seed for each attempt)

        while accepted_count < num_onpolicy_rollouts:
            print(f"----------Attempt {attempt_idx} (accepted {accepted_count})------------------")
            ts = 0
            count_total_states = 0
            count_attacked_states = 0
            curr_attacked_state_idxs = []
            attempt_seed = seed + attempt_idx
            env = make_env(attempt_seed)
            obs, info = env.reset(seed=attempt_seed)  #

            cum_reward = 0
            Gaussian_random_sampled_attack_idxs = np.random.choice(a=fixed_rollout_len, size=budget, replace=False)
            Gaussian_random_sampled_attack_idxs = np.sort(Gaussian_random_sampled_attack_idxs).tolist()
            print(
                f'To be Gaussian attacked states are {len(Gaussian_random_sampled_attack_idxs)} with idxs :{Gaussian_random_sampled_attack_idxs}')
            while ts < fixed_rollout_len:
                unnormalized_obs = obs
                if ts in Gaussian_random_sampled_attack_idxs:
                    '''add attack'''
                    attacked_obs = add_adv_gaussian_patch_Deterministic(unnormalized_obs)
                    attacked_obs = attacked_obs/255.0
                    
                    count_attacked_states += 1
                    if attacked_obs.ndim == 3:
                        attacked_obs = np.transpose(attacked_obs, (2, 0, 1))  # HWC → CHW
                        attacked_obs = torch.tensor(attacked_obs, dtype=torch.float32).unsqueeze(0).to(device)
                    action = backdoor_policy.forward(attacked_obs)[0].detach().cpu().numpy().astype(np.float32)
                    obs, reward, terminated, truncated, _ = env.step(action)
                    curr_attacked_state_idxs.append(ts)
                    # print(f"True pred action {curr_pred_action} but taking test time trigger action {curr_at/tacked_action} at ts {ts} has reward {reward}")
                else:
                    obs_nt = np.transpose((unnormalized_obs/255.0),(2,0,1))
                    obs_t = torch.tensor(obs_nt, dtype=torch.float32).unsqueeze(0).to(device)
                    action = backdoor_policy.forward(obs_t)
                    action = action[0].detach().cpu().numpy().astype(np.float32)
                    obs, reward, terminated,truncated, _ = env.step(action)
                    # obs, reward, terminated, truncated, _ = env.step(curr_pred_action)
                    # print(f"True pred action {curr_pred_action} at ts {ts} has reward {reward}")

                cum_reward += reward
                ts += 1
                count_total_states += 1

            # if count_total_states == fixed_rollout_len and truncated == True:
            if (count_total_states == fixed_rollout_len) and (truncated is True) and (
                        count_attacked_states == budget):

                print( f'Finished Gaussian Randomattacking {count_attacked_states} states reached episode len{count_total_states} and Truncated{truncated}')
                ep_idx = accepted_count  # index in the kept-rollouts dicts (0..num_onpolicy_rollouts-1)
                count_total_states_dict[ep_idx] = count_total_states
                count_attacked_state_dict[ep_idx] = count_attacked_states
                attacked_state_idx_dict[ep_idx] = curr_attacked_state_idxs
                reward_dict[ep_idx] = cum_reward
                episode_seed_dict[ep_idx] = attempt_seed
                print(f"Gaussian Random Attack Accepted rollout {ep_idx} (seed {attempt_seed}): Reward {cum_reward}, "
                    f"attacked {count_attacked_states}/{count_total_states} states.")
                accepted_count += 1
            else:
                if terminated:
                    print( f'Terminated early for attempt idx{attempt_idx} with attempt seed {attempt_seed} reached episode len{ts}')
            attempt_idx += 1
            env.close()
        print('--!!--' * 10)
        print(f"Gaussian Random Reward dict {reward_dict} with length of rollouts {count_total_states_dict} "
              f"and states attacked {count_attacked_state_dict}")
        return reward_dict, count_total_states_dict, count_attacked_state_dict, attacked_state_idx_dict, episode_seed_dict

def final_correct_seeded_gaussian_entropy_attack_with_budget_all_non_gas_fixed_len(backdoor_policy,entropy_threshold,num_onpolicy_rollouts=5,budget=300,fixed_rollout_len=1000):
    device = torch.device(
        "mps" if torch.backends.mps.is_available()
        else ("cuda" if torch.cuda.is_available() else "cpu"))
    backdoor_policy.to(device)
    backdoor_policy.eval()
    
    count_total_states_dict = defaultdict(list)
    count_attacked_state_dict = defaultdict(list)
    attacked_state_idx_dict = defaultdict(list)
    reward_dict = defaultdict(list)
    episode_seed_dict = defaultdict(int)
    
    accepted_count = 0
    attempt_idx = 0
    while accepted_count < num_onpolicy_rollouts:
        ts = 0
        count_total_states = 0
        count_attacked_states = 0
        curr_attacked_state_idxs = []
        attempt_seed = seed + attempt_idx
        print(
            f"----------Gaussian Entropy Attempting {attempt_idx} with seed {attempt_seed} and total accepted full len rollouts {accepted_count}------------------")
        env = make_env(attempt_seed)
        obs, info = env.reset(seed=attempt_seed)
        cum_reward = 0.0
        while ts < fixed_rollout_len:
            unnormalized_obs = obs
            obs_nt = np.transpose((unnormalized_obs/255.0),(2,0,1))
            obs_t = torch.tensor(obs_nt, dtype=torch.float32).unsqueeze(0).to(device)
            clean_action, ent = gaussian_action_and_entropy(backdoor_policy, obs_t)
            clean_is_target = is_target_action_continuous(clean_action)
            
            do_attack = (ent < entropy_threshold) and (count_attacked_states < budget) and (not clean_is_target)
            #     '''gas should not be optimal/near optimal'''
            # if curr_entropy < entropy_threshold and count_attacked_states<budget and  curr_pred_action!=3:
            if do_attack:
                # print(f"Current pred action: {curr_pred_action} and Entropy{curr_entropy}at ts {ts}")
                '''add attack'''
                attacked_obs = add_adv_gaussian_patch_Deterministic(unnormalized_obs)
                attacked_obs = attacked_obs/255.0
                # print(f"--------Attacking state at ts {ts} and current attacked state count {count_attacked_states}----------------")
                count_attacked_states += 1
                if attacked_obs.ndim == 3:
                    attacked_obs = np.transpose(attacked_obs, (2, 0, 1))  # HWC → CHW
                    attacked_obs = torch.tensor(attacked_obs, dtype=torch.float32).unsqueeze(0).to(device)
                attacked_action, _ = gaussian_action_and_entropy(backdoor_policy, attacked_obs)
                # action = backdoor_policy.forward(attacked_obs)[0].detach().cpu().numpy().astype(np.float32)
                obs, reward, terminated, truncated, info = env.step(attacked_action)
                curr_attacked_state_idxs.append(ts)
                # print(f"True pred action {curr_pred_action} but taking test time trigger action {curr_attacked_action} at ts {ts} has reward {reward}")
            else:
                obs, reward, terminated, truncated, info = env.step(clean_action)

            cum_reward += float(reward)
            ts += 1
            count_total_states += 1
        if (count_total_states == fixed_rollout_len) and (truncated is True) and (count_attacked_states == budget):

            ep_idx = accepted_count  # index in the kept-rollouts dicts (0..num_onpolicy_rollouts-1)
            count_total_states_dict[ep_idx] = count_total_states
            count_attacked_state_dict[ep_idx] = count_attacked_states
            attacked_state_idx_dict[ep_idx] = curr_attacked_state_idxs
            reward_dict[ep_idx] = cum_reward
            episode_seed_dict[ep_idx] = attempt_seed

            print(
                f"Accepted rollout {ep_idx} (seed {attempt_seed}): Reward {cum_reward}, "
                f"attacked {count_attacked_states}/{count_total_states} states.")
            accepted_count += 1
        else:
            if terminated:
                print(f'Terminated early for attempt idx{attempt_idx} with attempt seed {attempt_seed} reached episode len{ts}')
        attempt_idx += 1
        env.close()

    print('--!!--' * 10)
    print(f"Gaussian ENTROPY Reward dict {reward_dict} with length of rollouts {count_total_states_dict} and states attacked {count_attacked_state_dict}")
    return reward_dict, count_total_states_dict, count_attacked_state_dict, attacked_state_idx_dict, episode_seed_dict

def unattacked_eval_with_fixed_limit(backdoor_policy,num_onpolicy_rollouts,fixed_rollout_len=1000):
    print('----' * 15)
    print(f"Starting  Unattacked Eval")
    device = torch.device(
        "mps" if torch.backends.mps.is_available()
        else ("cuda" if torch.cuda.is_available() else "cpu"))
    backdoor_policy.to(device)
    backdoor_policy.eval()
    count_total_states_dict = defaultdict(list)

    reward_dict = defaultdict(list)
    episode_seed_dict = defaultdict(int)
    accepted_count = 0  # number of accepted (kept) rollouts
    attempt_idx = 0

    while accepted_count < num_onpolicy_rollouts:
        print(f"----------Attempt {attempt_idx} (accepted {accepted_count})------------------")
        ts = 0
        count_total_states = 0
        count_attacked_states = 0
        curr_attacked_state_idxs = []
        attempt_seed = seed + attempt_idx
        env = make_env(attempt_seed)
        obs, info = env.reset(seed=attempt_seed)  #
        cum_reward = 0
        while ts < fixed_rollout_len:
            unnormalized_obs = obs
            obs_nt = np.transpose((unnormalized_obs/255.0), (2,0,1))
            obs_t = torch.tensor(obs_nt, dtype=torch.float32).unsqueeze(0).to(device)
            action = backdoor_policy.forward(obs_t)
            action = action[0].detach().cpu().numpy().astype(np.float32)
            obs, reward, terminated,truncated, _ = env.step(action)
            # print(f"True pred action {curr_pred_action} at ts {ts} has reward {reward}")

            cum_reward += reward
            ts += 1
            count_total_states += 1

        if (count_total_states == fixed_rollout_len) and (truncated is True):

            print(f'Finished Unattacked Eval of backdoor policy reached episode len{count_total_states} and Truncated{truncated}')
            ep_idx = accepted_count  # index in the kept-rollouts dicts (0..num_onpolicy_rollouts-1)
            count_total_states_dict[ep_idx] = count_total_states
            reward_dict[ep_idx] = cum_reward
            episode_seed_dict[ep_idx] = attempt_seed
            print(f"Unattack Eval Accepted rollout {ep_idx} (seed {attempt_seed}): Reward {cum_reward}")
            accepted_count += 1
        else:
            if terminated:
                print(
                    f'Terminated early for attempt idx{attempt_idx} with attempt seed {attempt_seed} reached episode len{ts}')
        attempt_idx += 1
        env.close()

    print('--!!--' * 10)
    print(f"UnAttacked Eval Reward dict {reward_dict} with length of rollouts {count_total_states_dict}")
    return reward_dict, count_total_states_dict, episode_seed_dict

def save_results(save_dir, type_attack, reward_dict, total_states_dict, attacked_state_dict, attacked_idx_dict,episode_seed_dict):
    np.save(os.path.join(save_dir, f"{type_attack}_attack_reward_dict.npy"), reward_dict)
    np.save(os.path.join(save_dir, f"{type_attack}_attack_count_total_states_dict.npy"), total_states_dict)
    np.save(os.path.join(save_dir, f"{type_attack}_attack_count_attacked_state_dict.npy"), attacked_state_dict)
    np.save(os.path.join(save_dir, f"{type_attack}_attack_attacked_state_idx_dict.npy"), attacked_idx_dict)
    np.save(os.path.join(save_dir, f"{type_attack}_episode_seed_dict.npy"), episode_seed_dict)
    print(f"[INFO] Saved results for {type_attack} attack in {save_dir}")
def save_results_Unattacked(save_dir, patch_type, reward_dict, total_states_dict,episode_seed_dict):
    np.save(os.path.join(save_dir, f"{patch_type}_Unattack_reward_dict.npy"), reward_dict)
    np.save(os.path.join(save_dir, f"{patch_type}_Unattack_count_total_states_dict.npy"), total_states_dict)
    np.save(os.path.join(save_dir, f"{patch_type}_Unattack_episode_seed_dict.npy"), episode_seed_dict)
    print(f"[INFO] Saved results for {patch_type} Unattacked Eval in {save_dir}")

def create_exp_dir(attack_budget, entropy_threshold, seed,num_onpolicy_rollouts,seq_halves2=False):
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    # base_path= '/ethan/test_time_attack_runs/Fixed-TS-Fixed-Budget-Fixed-Sampling/'
    # if not os.path.exists(base_path):
    #     os.makedirs(base_path)
    base_path = os.path.abspath("./test_time_attack_runs/Fixed-TS-Fixed-Budget-Fixed-Sampling/")
    os.makedirs(base_path, exist_ok=True)
    # if not seq_halves2:
    dir_name = f"New-TTT-5-percent-poisoned-budget_{attack_budget}_entropy_{entropy_threshold}_base_seed_{seed}_num_onpolicy_rollouts{num_onpolicy_rollouts}-env-v3-time{timestamp}"
    curr_dir_path = os.path.join(base_path, dir_name)
    if not os.path.exists(curr_dir_path):
        os.makedirs(curr_dir_path, exist_ok=True)
    print(f"[INFO] Created experiment dir: {curr_dir_path}")
    return curr_dir_path
def print_results_mean_standard_error(num_on_policy_rollouts,
    eval_rewards_red, entropy_attack_reward_dict_red, random_attack_reward_dict_red,
    eval_rewards_gaussian, entropy_attack_reward_dict_gaussian, random_attack_reward_dict_gaussian,
    attack_budget, entropy_threshold):
    # Compute mean rewards
   
    eval_mean_reward_red = np.mean(list(eval_rewards_red.values()))
    entropy_mean_reward_red = np.mean(list(entropy_attack_reward_dict_red.values()))
    random_mean_reward_red = np.mean(list(random_attack_reward_dict_red.values()))

    
    eval_mean_reward_gaussian = np.mean(list(eval_rewards_gaussian.values()))
    entropy_mean_reward_gaussian = np.mean(list(entropy_attack_reward_dict_gaussian.values()))
    random_mean_reward_gaussian = np.mean(list(random_attack_reward_dict_gaussian.values()))

    # Compute standard error


    def std_err(values):
        return np.std(values) / np.sqrt(num_on_policy_rollouts) if len(values) > 1 else 0


    red_errors = [
        # std_err(eval_rewards_red),
        std_err(list(eval_rewards_red.values())),
        std_err(list(random_attack_reward_dict_red.values())),
        std_err(list(entropy_attack_reward_dict_red.values()))
    ]

    gaussian_errors = [
        # std_err(eval_rewards_gaussian),
        std_err(list(eval_rewards_gaussian.values())),
        std_err(list(random_attack_reward_dict_gaussian.values())),
        std_err(list(entropy_attack_reward_dict_gaussian.values())) ]

    print(f'Unattacked Mean for Red {np.round(eval_mean_reward_red,3)}, standard error {np.round(red_errors[0],3)}')
    print(f'Random Attack Mean for Red {np.round(random_mean_reward_red,3)}, standard error {np.round(red_errors[1],3)}')
    print(f'Entropy Attack Mean for Red {np.round(entropy_mean_reward_red,3)}, standard error {np.round(red_errors[2],3)}')
    print('--'*15)
    print(f'Unattacked Mean for Gaussian {np.round(eval_mean_reward_gaussian,3)}, standard error {np.round(gaussian_errors[0],3)}')
    print(f'Random Attack Mean for Gaussian {np.round(random_mean_reward_gaussian,3)}, standard error {np.round(gaussian_errors[1],3)}')
    print(f'Entropy Attack Mean for Gaussian {np.round(entropy_mean_reward_gaussian,3)}, standard error {np.round(gaussian_errors[2],3)}')

if __name__ == '__main__':
    model_to_test = "BC_P_0_SEED_0.pt"  # Change to "BC_P_0_SEED_0.pt" for the clean baseline
    
    model_folder = "../models/BC_red1_cameraready_run2/"
    full_model_path = os.path.join(model_folder, model_to_test)
    
    device = torch.device("mps" if torch.backends.mps.is_available() 
                          else ("cuda" if torch.cuda.is_available() else "cpu"))

    # Loading Agent
    print(f"\n" + "="*50)
    print(f"LOADING MODEL: {full_model_path}")
    print("="*50)
    
    if not os.path.exists(full_model_path):
        print(f"ERROR: File not found at {full_model_path}")
        print("Please verify your folder structure matches the screenshot.")
        exit()

    agent = PolicyNetwork()
    agent.load_state_dict(torch.load(full_model_path, weights_only=True, map_location=device))
    agent.to(device)
    agent.eval()

    # Setting up exp directory
    # Keeping rollouts low for quick iter
    demo_rollouts = 3 
    curr_dir = create_exp_dir(attack_budget=attack_budget, 
                              entropy_threshold=entropy_threshold, 
                              seed=seed,
                              num_onpolicy_rollouts=demo_rollouts)

    # Running entropy attack
    with torch.no_grad():
        print(f"Starting Entropy-Based Red Patch Attack...")
        print(f"Targeting: {model_to_test} | Budget: {attack_budget} | Threshold: {entropy_threshold}")
        
        results = final_correct_seeded_red_entropy_attack_with_budget_all_non_gas_ensure_fixed_len(
            agent, 
            entropy_threshold=entropy_threshold, 
            num_onpolicy_rollouts=demo_rollouts,
            budget=attack_budget,
            fixed_rollout_len=max_rollout_len
        )
        
        # Unpacking  results and saving to .npy
        reward_dict, total_states, attacked_states, attacked_idxs, episode_seeds = results
        save_results(curr_dir, "red_entropy_demo", 
                     reward_dict, total_states, attacked_states, attacked_idxs, episode_seeds)

    # Summary Output
    avg_reward = np.mean(list(reward_dict.values()))
    print("\n" + "="*50)
    print(f"DEMO COMPLETE")
    print(f"Model: {model_to_test}")
    print(f"Average Reward: {avg_reward:.2f}")
    print(f"Results saved to: {curr_dir}")
    print("="*50)

# if __name__ == '__main__':
#       agent = PolicyNetwork()
#       agent.load_state_dict(torch.load(red_BC_policy_path, weights_only=True))
#       curr_dir = create_exp_dir(attack_budget=attack_budget, entropy_threshold=entropy_threshold, seed=seed,
#                                 num_onpolicy_rollouts=num_onpolicy_rollouts, )

#       random_attack_reward_dict, random_attack_count_total_states_dict, random_attack_count_attacked_state_dict, random_attack_attacked_state_idx_dict, red_random_epsiode_seed_dict = final_correct_seeded_red_random_attack_with_budget_fixed_len_changed_random_sampling(
#           agent, num_onpolicy_rollouts=num_onpolicy_rollouts, budget=attack_budget, fixed_rollout_len=max_rollout_len)
#       save_results(curr_dir, "random_red", random_attack_reward_dict, random_attack_count_total_states_dict,
#                    random_attack_count_attacked_state_dict, random_attack_attacked_state_idx_dict,
#                    red_random_epsiode_seed_dict)

#       entropy_attack_reward_dict, entropy_attack_count_total_states_dict, entropy_attack_count_attacked_state_dict, entropy_attack_attacked_state_idx_dict, red_entropy_episode_seed_dict = final_correct_seeded_red_entropy_attack_with_budget_all_non_gas_ensure_fixed_len(
#           agent, entropy_threshold=entropy_threshold, num_onpolicy_rollouts=num_onpolicy_rollouts,
#           budget=attack_budget,
#           fixed_rollout_len=max_rollout_len)
#       save_results(curr_dir, "entropy_red", entropy_attack_reward_dict, entropy_attack_count_total_states_dict,
#                    entropy_attack_count_attacked_state_dict, entropy_attack_attacked_state_idx_dict,
#                    red_entropy_episode_seed_dict)

#       gaussian_agent = PolicyNetwork()
#       gaussian_agent.load_state_dict(torch.load(gaussian_BC_policy_path, weights_only=True))
     

#       gaussian_random_attack_reward_dict, gaussian_random_attack_count_total_states_dict, gaussian_random_attack_count_attacked_state_dict, gaussian_random_attack_attacked_state_idx_dict, gaussian_random_epsiode_seed_dict = final_correct_seeded_gaussian_random_attack_with_budget_fixed_len_changed_random_sampling(
#           gaussian_agent, num_onpolicy_rollouts=num_onpolicy_rollouts, budget=attack_budget,
#           fixed_rollout_len=max_rollout_len)
#       save_results(curr_dir, "random_gaussian", gaussian_random_attack_reward_dict,
#                    gaussian_random_attack_count_total_states_dict,
#                    gaussian_random_attack_count_attacked_state_dict, gaussian_random_attack_attacked_state_idx_dict,
#                    gaussian_random_epsiode_seed_dict)

#       gaussian_entropy_attack_reward_dict, gaussian_entropy_attack_count_total_states_dict, gaussian_entropy_attack_count_attacked_state_dict, gaussian_entropy_attack_attacked_state_idx_dict, gaussian_entropy_epsiode_seed_dict = final_correct_seeded_gaussian_entropy_attack_with_budget_all_non_gas_fixed_len(
#           gaussian_agent, entropy_threshold=entropy_threshold, num_onpolicy_rollouts=num_onpolicy_rollouts,
#           budget=attack_budget,
#           fixed_rollout_len=max_rollout_len)

#       save_results(curr_dir, "entropy_gaussian", gaussian_entropy_attack_reward_dict,
#                    gaussian_entropy_attack_count_total_states_dict,
#                    gaussian_entropy_attack_count_attacked_state_dict, gaussian_entropy_attack_attacked_state_idx_dict,
#                    gaussian_entropy_epsiode_seed_dict)

#       gaussian_Unattack_reward_dict, gaussian_Unattack_count_total_states_dict, gaussian_Unattack_epsiode_seed_dict = unattacked_eval_with_fixed_limit(
#           gaussian_agent, num_onpolicy_rollouts=num_onpolicy_rollouts,
#           fixed_rollout_len=max_rollout_len)
#       save_results_Unattacked(curr_dir, "gaussian", gaussian_Unattack_reward_dict,
#                               gaussian_Unattack_count_total_states_dict, gaussian_Unattack_epsiode_seed_dict)
#       red_Unattack_reward_dict, red_Unattack_count_total_states_dict, red_Unattack_epsiode_seed_dict = unattacked_eval_with_fixed_limit(
#           agent, num_onpolicy_rollouts=num_onpolicy_rollouts,
#           fixed_rollout_len=max_rollout_len)
#       save_results_Unattacked(curr_dir, "red", red_Unattack_reward_dict,
#                               red_Unattack_count_total_states_dict, red_Unattack_epsiode_seed_dict)
#       print_results_mean_standard_error(
#           num_onpolicy_rollouts,
#           red_Unattack_reward_dict, entropy_attack_reward_dict, random_attack_reward_dict,
#           gaussian_Unattack_reward_dict, gaussian_entropy_attack_reward_dict, gaussian_random_attack_reward_dict,
#           attack_budget=attack_budget, entropy_threshold=entropy_threshold
#       )

