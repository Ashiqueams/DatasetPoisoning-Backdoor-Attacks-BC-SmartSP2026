# Run commands — run54 (with/without budget) and run51

All commands assume `cwd` = repo root, and use `PYTHONPATH=.` so the scripts can
import sibling modules (`policyNetwork_nll`, etc.).

## Legend

- **run54_steer_clean_adv** — trigger + PGD (the main attack). `poison_levels: [0,25,50,75,100]`, `model_seeds: [0,1,2,3,4]`.
- **run51_steer_clean_right** — trigger ONLY, no PGD (the ablation baseline for run54 — see config.yaml comments "isolating the PGD step as the only variable"). `poison_levels: [0,100]`, `model_seeds: [0]` only.
- **"without budget"** = `--budget 0`. With `budget=0`, the gate's `count_attacked < budget` check (`0 < 0`) is always False, so the attack never fires — this gives a pure no-attack control point, run through the *exact same* harness (same seeds, same episode length) as the attacked runs, which is a cleaner comparison than relying on the separately-printed `clean_control` numbers.
- Entropy thresholds are loaded from `entropy_thresholds_steer_{run}_P{poison_level}.npy` (see `THRESHOLDS_PATH` in `ttt_steer_entropy.py`) — this file must exist and contain an entry for whatever `--budget` you pass, or the script will KeyError.

## ⚠ Known data issues — check before trusting new numbers

- `entropy_thresholds_steer_run54_steer_clean_adv_P25.npy` is currently **broken**: every budget for every seed is exactly `0.0` (see conversation — likely zero eligible frames found during that calibration run). **Recalibrate before using P=25.**
- No threshold file exists yet for **P=50** or **P=75** on run54 — calibration has never been run for these levels.
- No threshold file exists yet for **run51** at all — calibration has never been run for this run.
- Only P=100 for run54 currently has a real, full-budget-sweep calibration (`entropy_thresholds_steer_run54_steer_clean_adv_P100.npy`, keys `{50,100}` — the file the script actually reads — plus an archived `_all_budgets` copy with keys `{5,10,15,50,100,250,500,1000}`).

## Step 1 — calibrate entropy thresholds (`entropy/entropy_steer.py`)

`entropy/entropy_steer.py` has a **hardcoded constant** (line 35): `BUDGETS = [25, 50]`.
To get thresholds for the full budget sweep used below (`0,5,10,15,50,100,250,500,1000`),
edit that line locally first:

```python
BUDGETS = [5, 10, 15, 50, 100, 250, 500, 1000]
```

Then run calibration for each `(run, poison_level)` pair that needs it. This overwrites
`entropy_thresholds_steer_{run}_P{poison_level}.npy` — the exact file `ttt_steer_entropy.py`
reads — so no renaming is needed. Optionally `cp` a dated backup if you want to keep the
old thresholds around for comparison.

```bash
# --- run54, P=25 (BROKEN — rerun) ---
PYTHONPATH=. python entropy/entropy_steer.py --run run54_steer_clean_adv --poison_level 25 \
    --gate_direction hard_left --steer_bound -0.3
# sanity check before trusting it: the script prints "<n> eligible (<pct>%)" per seed — if n=0, the bug is confirmed and needs investigating (not just rerunning).

# --- run54, P=50 (never calibrated) ---
PYTHONPATH=. python entropy/entropy_steer.py --run run54_steer_clean_adv --poison_level 50 \
    --gate_direction hard_left --steer_bound -0.3

# --- run54, P=75 (never calibrated) ---
PYTHONPATH=. python entropy/entropy_steer.py --run run54_steer_clean_adv --poison_level 75 \
    --gate_direction hard_left --steer_bound -0.3

# --- run54, P=100 (already calibrated — only rerun if you want the full 8-budget set in the live file, not just {50,100}) ---
PYTHONPATH=. python entropy/entropy_steer.py --run run54_steer_clean_adv --poison_level 100 \
    --gate_direction hard_left --steer_bound -0.3

# --- run51, P=100 (ablation baseline — never calibrated; single model seed) ---
PYTHONPATH=. python entropy/entropy_steer.py --run run51_steer_clean_right --poison_level 100 \
    --model_seeds 0 --gate_direction hard_left --steer_bound -0.3
```

## Step 2 — attack sweep, with and without budget (`ttt_attacks_script/ttt_steer_entropy.py`)

`--budget 0` is the no-attack control point in each loop. `--num_rollouts 80` matches the
scale already reported in `reward_cliff.html` (5 seeds × 80 rollouts = n=400) — this is a
lot of full CarRacing episodes (up to 1000 steps each) × 5 model seeds × 9 budgets, so for
a first sanity pass consider `--num_rollouts 10` before committing to the full run.

`--model_seeds` is omitted below so each run loops over every seed in `config.yaml` for
that run automatically (`[0,1,2,3,4]` for run54, `[0]` for run51).

```bash
# ============ run54 (trigger + PGD) ============

# -- P=25 (after recalibrating in Step 1) --
for b in 0 5 10 15 50 100 250 500 1000; do
  PYTHONPATH=. python ttt_attacks_script/ttt_steer_entropy.py \
      --run run54_steer_clean_adv --poison_level 25 \
      --gate_direction hard_left --steer_bound -0.3 \
      --sustain_len 1 --warmup_steps 60 --num_rollouts 80 --budget $b
done

# -- P=50 (after calibrating in Step 1) --
for b in 0 5 10 15 50 100 250 500 1000; do
  PYTHONPATH=. python ttt_attacks_script/ttt_steer_entropy.py \
      --run run54_steer_clean_adv --poison_level 50 \
      --gate_direction hard_left --steer_bound -0.3 \
      --sustain_len 1 --warmup_steps 60 --num_rollouts 80 --budget $b
done

# -- P=75 (after calibrating in Step 1) --
for b in 0 5 10 15 50 100 250 500 1000; do
  PYTHONPATH=. python ttt_attacks_script/ttt_steer_entropy.py \
      --run run54_steer_clean_adv --poison_level 75 \
      --gate_direction hard_left --steer_bound -0.3 \
      --sustain_len 1 --warmup_steps 60 --num_rollouts 80 --budget $b
done

# -- P=100 (already calibrated for budgets {50,100}; rerun Step 1 first for the full sweep) --
for b in 0 5 10 15 50 100 250 500 1000; do
  PYTHONPATH=. python ttt_attacks_script/ttt_steer_entropy.py \
      --run run54_steer_clean_adv --poison_level 100 \
      --gate_direction hard_left --steer_bound -0.3 \
      --sustain_len 1 --warmup_steps 60 --num_rollouts 80 --budget $b
done

# ============ run51 (trigger only — PGD ablation baseline) ============
# single model seed (0), poison_levels [0, 100] only per config.yaml

for b in 0 5 10 15 50 100 250 500 1000; do
  PYTHONPATH=. python ttt_attacks_script/ttt_steer_entropy.py \
      --run run51_steer_clean_right --poison_level 100 --model_seeds 0 \
      --gate_direction hard_left --steer_bound -0.3 \
      --sustain_len 1 --warmup_steps 60 --num_rollouts 80 --budget $b
done

# optional sanity check: attack mechanics on the CLEAN (P=0) model should do ~nothing
# regardless of run, since there's no backdoor to trigger. Needs its own calibration first
# (entropy_steer.py --poison_level 0 for that run) since thresholds are model-specific.
```

## Where results land

Each invocation writes to `./ttt_steer_entropy_runs/{run}/P{level}_seed0_time<timestamp>/model_seed{N}/`:
- `clean_control_rewards.npy`, `clean_control_outcomes.npy`, `clean_control_tile_fracs.npy`
- `steer_entropy_attack_rewards.npy`, `steer_entropy_attack_attacked_counts.npy`, `steer_entropy_attack_outcomes.npy`, `steer_entropy_attack_tile_fracs.npy`

These are what to load for the reward-vs-budget comparison plot (same shape of data as
behind `reward_cliff.html`, just extended across poison levels and with run51 added).
