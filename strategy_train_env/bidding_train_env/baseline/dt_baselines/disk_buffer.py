import ast
import hashlib
import json
import os
import pickle
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
from bidding_train_env.baseline.dt_baselines.utils import EpisodeReplayBuffer


class DiskReplayBuffer(EpisodeReplayBuffer):
    """Chunked CSV conversion; only one episode is materialized per lookup."""
    def __init__(self, data_path, cache_dir, state_dim=16, act_dim=1, K=10,
                 max_ep_len=48, scale=2000, chunksize=10000):
        paths = [Path(p).resolve() for p in ([data_path] if isinstance(data_path, str) else data_path)]
        signature = [(str(p), p.stat().st_size, p.stat().st_mtime_ns) for p in paths]
        key = hashlib.sha256(json.dumps([2, signature, state_dim]).encode()).hexdigest()[:20]
        os.makedirs(cache_dir, exist_ok=True)
        cache = Path(cache_dir) / (key + '.sqlite')
        self.device, self.state_dim, self.act_dim = 'cpu', state_dim, act_dim
        self.K, self.max_ep_len, self.scale = K, max_ep_len, scale
        if not cache.exists():
            temporary = cache.with_suffix('.building')
            if temporary.exists():
                temporary.unlink()
            db = sqlite3.connect(str(temporary))
            db.execute('CREATE TABLE episodes (id INTEGER PRIMARY KEY, data BLOB)')
            count, total = 0, 0
            mean, m2 = np.zeros(state_dim), np.zeros(state_dim)
            lengths, pending = [], []
            columns = ['state', 'action', 'reward', 'done', 'budget', 'CPAConstraint', 'realAllCost']
            for path in paths:
                for chunk in pd.read_csv(path, usecols=columns, chunksize=chunksize):
                    for row in chunk.to_dict('records'):
                        row['state'] = np.asarray(ast.literal_eval(row['state']), dtype=np.float64)
                        if row['state'].shape != (state_dim,) or not np.isfinite(row['state']).all():
                            raise ValueError(f'Invalid state in {path}')
                        pending.append(row)
                        if len(pending) > max_ep_len:
                            raise ValueError(f'Episode exceeds {max_ep_len} rows; check ordering/done in {path}')
                        if not row['done']:
                            continue
                        if len(pending) > 1:
                            states = np.stack([r['state'] for r in pending])
                            costs = [(r['state'][1] - pending[i+1]['state'][1]) * r['budget']
                                     for i, r in enumerate(pending[:-1])]
                            costs.append(row['realAllCost'] - (1-row['state'][1])*row['budget'])
                            traj = dict(observations=states, actions=np.array([r['action'] for r in pending])[:, None],
                                        rewards=np.array([r['reward'] for r in pending])[:, None],
                                        dones=np.array([r['done'] for r in pending]), cost_ts=np.array(costs),
                                        budget=row['budget'], cpa_constrain=row['CPAConstraint'])
                            db.execute('INSERT INTO episodes VALUES (?, ?)', (count, pickle.dumps(traj, protocol=4)))
                            n = len(states)
                            delta = states.mean(0) - mean
                            m2 += ((states-states.mean(0))**2).sum(0) + delta**2 * total*n/(total+n)
                            mean += delta*n/(total+n)
                            total += n
                            lengths.append(n)
                            count += 1
                        pending = []
                    db.commit()
            if pending:
                raise ValueError('Training data ends with an unfinished episode')
            if not count:
                raise ValueError('No complete training episodes')
            metadata = dict(mean=mean, std=np.maximum(np.sqrt(m2/total), 1e-6), lengths=np.array(lengths))
            db.execute('CREATE TABLE metadata (data BLOB)')
            db.execute('INSERT INTO metadata VALUES (?)', (pickle.dumps(metadata),))
            db.commit()
            db.close()
            temporary.replace(cache)
        self.db = sqlite3.connect(str(cache))
        self.db.execute('PRAGMA cache_size=-8192')
        metadata = pickle.loads(self.db.execute('SELECT data FROM metadata').fetchone()[0])
        self.state_mean, self.state_std = metadata['mean'], metadata['std']
        self.traj_lens = metadata['lengths']
        self.sorted_inds = np.arange(len(self.traj_lens))
        self.p_sample = self.traj_lens / self.traj_lens.sum()
        self.trajectories = EpisodeTable(self)

    def __len__(self):
        return len(self.traj_lens)

    def __getitem__(self, index):
        # Parent constructs a normalized, padded training window.
        return super().__getitem__(index)

    def close(self):
        self.db.close()


class EpisodeTable:
    def __init__(self, buffer):
        self.buffer = buffer

    def __len__(self):
        return len(self.buffer)

    def __getitem__(self, index):
        return pickle.loads(self.buffer.db.execute('SELECT data FROM episodes WHERE id=?', (int(index),)).fetchone()[0])
