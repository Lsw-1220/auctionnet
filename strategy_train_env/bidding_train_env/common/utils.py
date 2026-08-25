import pandas as pd
import os
import pickle
import numpy as np
import glob
from copy import deepcopy
import torch
import ast


def resolve_training_data_paths(data_paths):
    """Resolve CSV inputs from files, directories, comma-separated values, or globs."""
    if isinstance(data_paths, (str, os.PathLike)):
        inputs = [item.strip() for item in str(data_paths).split(',') if item.strip()]
    else:
        inputs = [str(item).strip() for item in data_paths if str(item).strip()]

    resolved = []
    for item in inputs:
        if os.path.isdir(item):
            matches = glob.glob(os.path.join(item, '*.csv'))
        else:
            matches = glob.glob(item)
        if not matches:
            raise FileNotFoundError(f'No training CSV matched: {item}')
        resolved.extend(path for path in sorted(matches) if path.lower().endswith('.csv'))

    # Preserve user/glob order while avoiding accidental duplicate loading.
    resolved = list(dict.fromkeys(resolved))
    if not resolved:
        raise FileNotFoundError(f'No training CSV files found in: {data_paths}')
    return resolved


def load_training_csvs(data_paths):
    """Load and concatenate all CSV inputs into one training DataFrame."""
    paths = resolve_training_data_paths(data_paths)
    return pd.concat([pd.read_csv(path) for path in paths], ignore_index=True), paths


def _parse_vector_column(series, state_dim):
    """Parse a CSV column of list-like vectors into one contiguous float32 array."""
    output = np.empty((len(series), state_dim), dtype=np.float32)
    for row_index, value in enumerate(series):
        try:
            if isinstance(value, str):
                # The normal CSV representation is a flat numeric list. NumPy's
                # parser is substantially faster than constructing Python float
                # objects with literal_eval for tens of millions of rows.
                vector = np.fromstring(value.strip().strip("[]()"), sep=",", dtype=np.float32)
                if vector.size != state_dim:
                    vector = np.asarray(ast.literal_eval(value), dtype=np.float32)
            else:
                vector = np.asarray(value, dtype=np.float32)
            if vector.size != state_dim or not np.isfinite(vector).all():
                output[row_index] = 0.0
            else:
                output[row_index] = vector.reshape(state_dim)
        except (ValueError, SyntaxError, TypeError):
            output[row_index] = 0.0
    return output


def load_compact_transition_arrays(data_paths, state_dim=16, normalize_indices=(13, 14, 15),
                                   reward_column="reward_continuous", chunksize=200_000):
    """Load offline-RL transitions without retaining a giant object DataFrame.

    The CSV files are scanned twice. The first pass computes the same global
    normalization statistics as ``normalize_state``/``normalize_reward``; the
    second writes directly into contiguous float32 arrays.
    """
    paths = resolve_training_data_paths(data_paths)
    count = 0
    reward_min, reward_max = np.inf, -np.inf
    sums = np.zeros(state_dim, dtype=np.float64)
    sum_squares = np.zeros(state_dim, dtype=np.float64)
    state_min = np.full(state_dim, np.inf, dtype=np.float64)
    state_max = np.full(state_dim, -np.inf, dtype=np.float64)

    first_pass_columns = ["state", reward_column]
    for path in paths:
        for chunk in pd.read_csv(path, usecols=first_pass_columns, chunksize=chunksize):
            states = _parse_vector_column(chunk["state"], state_dim)
            rewards = chunk[reward_column].to_numpy(dtype=np.float64, copy=False)
            count += len(chunk)
            sums += states.sum(axis=0, dtype=np.float64)
            sum_squares += np.square(states, dtype=np.float64).sum(axis=0)
            state_min = np.minimum(state_min, states.min(axis=0))
            state_max = np.maximum(state_max, states.max(axis=0))
            reward_min = min(reward_min, float(np.nanmin(rewards)))
            reward_max = max(reward_max, float(np.nanmax(rewards)))

    if count == 0:
        raise ValueError("No transitions found in the training CSV files")

    means = sums / count
    # pandas.Series.std() uses the sample standard deviation (ddof=1).
    variances = (sum_squares - count * means * means) / max(count - 1, 1)
    stds = np.sqrt(np.maximum(variances, 0.0))
    normalize_dict = {
        i: {"min": state_min[i], "max": state_max[i], "mean": means[i], "std": stds[i]}
        for i in normalize_indices
    }

    states = np.empty((count, state_dim), dtype=np.float32)
    next_states = np.empty((count, state_dim), dtype=np.float32)
    actions = np.empty((count, 1), dtype=np.float32)
    rewards = np.empty((count, 1), dtype=np.float32)
    dones = np.empty((count, 1), dtype=np.float32)
    reward_range = reward_max - reward_min + 1e-8

    offset = 0
    columns = ["state", "action", reward_column, "next_state", "done"]
    for path in paths:
        for chunk in pd.read_csv(path, usecols=columns, chunksize=chunksize):
            size = len(chunk)
            selection = slice(offset, offset + size)
            state_chunk = _parse_vector_column(chunk["state"], state_dim)
            next_state_chunk = _parse_vector_column(chunk["next_state"], state_dim)
            for index in normalize_indices:
                denominator = state_max[index] - state_min[index] + 0.01
                state_chunk[:, index] = (state_chunk[:, index] - state_min[index]) / denominator
                next_state_chunk[:, index] = (next_state_chunk[:, index] - state_min[index]) / denominator
            done_chunk = chunk["done"].to_numpy(dtype=np.float32, copy=False)
            next_state_chunk[done_chunk == 1] = 0.0

            states[selection] = state_chunk
            next_states[selection] = next_state_chunk
            actions[selection, 0] = chunk["action"].to_numpy(dtype=np.float32, copy=False)
            rewards[selection, 0] = (
                chunk[reward_column].to_numpy(dtype=np.float32, copy=False) - reward_min
            ) / reward_range
            dones[selection, 0] = done_chunk
            offset += size

    return (states, actions, rewards, next_states, dones), normalize_dict, paths


def save_training_checkpoint(model, save_dir, step, normalize_dict, method='save_jit'):
    """Save a self-contained checkpoint without moving the training model to CPU."""
    checkpoint_dir = os.path.join(save_dir, f'checkpoint_{step:08d}')
    save_normalize_dict(normalize_dict, checkpoint_dir)
    base_model = model.module if isinstance(model, torch.nn.DataParallel) else model
    snapshot = deepcopy(base_model)
    getattr(snapshot, method)(checkpoint_dir)
    return checkpoint_dir


def normalize_state(training_data, state_dim, normalize_indices):
    """
    Normalize features for reinforcement learning.
    Args:
        training_data: A DataFrame containing the training data.
        state_dim: The total dimension of the features.
        normalize_indices: A list of indices of the features to be normalized.

    Returns:
        A dictionary containing the normalization statistics.
    """
    state_columns = [f'state{i}' for i in range(state_dim)]
    next_state_columns = [f'next_state{i}' for i in range(state_dim)]

    for i, (state_col, next_state_col) in enumerate(zip(state_columns, next_state_columns)):
        training_data[state_col] = training_data['state'].apply(
            lambda x: x[i] if x is not None and not np.isnan(x).any() else 0.0)
        training_data[next_state_col] = training_data['next_state'].apply(
            lambda x: x[i] if x is not None and not np.isnan(x).any() else 0.0)

    stats = {
        i: {
            'min': training_data[state_columns[i]].min(),
            'max': training_data[state_columns[i]].max(),
            'mean': training_data[state_columns[i]].mean(),
            'std': training_data[state_columns[i]].std()
        }
        for i in normalize_indices
    }

    for state_col, next_state_col in zip(state_columns, next_state_columns):
        if int(state_col.replace('state', '')) in normalize_indices:
            min_val = stats[int(state_col.replace('state', ''))]['min']
            max_val = stats[int(state_col.replace('state', ''))]['max']
            training_data[f'normalize_{state_col}'] = (
                                                              training_data[state_col] - min_val) / (
                                                              max_val - min_val + 0.01)
            training_data[f'normalize_{next_state_col}'] = (
                                                                   training_data[next_state_col] - min_val) / (
                                                                   max_val - min_val + 0.01)
            # 0.01 error too large?
        else:
            training_data[f'normalize_{state_col}'] = training_data[state_col]
            training_data[f'normalize_{next_state_col}'] = training_data[next_state_col]

    training_data['normalize_state'] = training_data.apply(
        lambda row: tuple(row[f'normalize_{state_col}'] for state_col in state_columns), axis=1)
    training_data['normalize_nextstate'] = training_data.apply(
        lambda row: tuple(row[f'normalize_{next_state_col}'] for next_state_col in next_state_columns), axis=1)

    return stats


def normalize_reward(training_data, reward_type):
    """
    Normalize rewards for reinforcement learning.

    Args:
        training_data: A DataFrame containing the training data.
        reward_type: reward:sparse reward   reward_continuous: continuous reward

    Returns:
        A Series of normalized rewards.
    """
    reward_range = training_data[reward_type].max() - training_data[reward_type].min() + 0.00000001
    training_data["normalize_reward"] = (
                                                training_data[reward_type] - training_data[
                                            reward_type].min()) / reward_range
    return training_data["normalize_reward"]


def save_normalize_dict(normalize_dict, save_dir):
    """
    Save the normalization dictionary to a Pickle file.

    Args:
        normalize_dict: The dictionary containing normalization statistics.
        save_dir: The directory to save the normalization dictionary.
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    save_path = os.path.join(save_dir, 'normalize_dict.pkl')
    with open(save_path, 'wb') as file:
        pickle.dump(normalize_dict, file)


if __name__ == '__main__':
    test_data = {
        'state': [(1, 2, 3), (4, 5, 6), (7, 8, 9)],
        'next_state': [(2, 3, 4), (5, 6, 7), (8, 9, 10)],
        'reward': [10, 20, 30]
    }
    training_data = pd.DataFrame(test_data)
    state_dim = 3
    normalize_indices = [0, 2]
    stats = normalize_state(training_data, state_dim, normalize_indices)
    normalize_reward(training_data)
    print(training_data)
    print(stats)
