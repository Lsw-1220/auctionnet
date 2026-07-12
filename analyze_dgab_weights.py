"""
Analyze DGAB model weights to understand what the model focuses on.
No forward pass needed — pure weight analysis.

Usage:
    python analyze_dgab_weights.py
"""

import sys, os, pickle
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

_PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_PROJECT_ROOT, 'strategy_train_env'))
# Put autobidding LAST so its bidding_train_env doesn't shadow AuctionNet's,
# but import DGAB module via its absolute path with autobidding prepended temporarily
_ab_root = 'D:/research/Experiment/autobidding'
_saved_path = list(sys.path)
sys.path.insert(0, _ab_root)  # needed for blocks.py import inside model_po
from bidding_train_env.baseline.dgab.model_po import DGAB
sys.path[:] = _saved_path  # restore

CKPT_DIR = 'D:/research/Experiment/autobidding/saved_model/dgab_400k_sparse'
DEVICE = 'cpu'
OUTPUT_DIR = os.path.join(_PROJECT_ROOT, 'exp_data', 'dgab_analysis')
os.makedirs(OUTPUT_DIR, exist_ok=True)

BLOCK_CONFIG = {
    'n_ctx': 1024, 'n_embd': 512, 'n_layer': 8, 'n_head': 16,
    'n_inner': 1024, 'activation_function': 'relu', 'n_position': 1024,
    'resid_pdrop': 0.1, 'attn_pdrop': 0.1,
}

print('Loading model...')
with open(os.path.join(CKPT_DIR, 'normalize_dict.pkl'), 'rb') as f:
    nd = pickle.load(f)

model = DGAB(
    base_state_dim=16, act_dim=1,
    hidden_size=512, max_ep_len=96, time_dim=8,
    block_config=BLOCK_CONFIG, actor_type='stack', critic_type='sequence',
    device=DEVICE,
)
model.load_state_dict(torch.load(os.path.join(CKPT_DIR, 'complete_train.pt'), map_location=DEVICE))
model.to(DEVICE)
model.eval()

actor = model.actor

# ── 1. State Embedding: per-dimension importance ──
print('\n' + '='*70)
print('1. STATE DIMENSION IMPORTANCE (embed_state weight column norms)')
print('='*70)

state_embed = actor.embed_state.weight.data.cpu().numpy()  # [512, 16]
state_col_norms = np.linalg.norm(state_embed, axis=0)

state_names = [
    'time_left', 'budget_left',          # 0,1
    'bid_mean', 'bid_tail',              # 2,3
    'lwc_mean', 'pv_mean', 'conv_mean', 'xi_mean',  # 4,5,6,7
    'lwc_tail', 'pv_tail', 'conv_tail', 'xi_tail',  # 8,9,10,11
    'cur_pv_mean', 'cur_vol', 'tail_vol', 'hist_vol',  # 12,13,14,15
]

print(f'  {"Dim":<15} {"Importance":>12} {"Bar"}')
print(f'  {"-"*50}')
for i in np.argsort(-state_col_norms):
    bar = '█' * int(state_col_norms[i] / state_col_norms.max() * 40)
    print(f'  {state_names[i]:<15} {state_col_norms[i]:>12.4f}  {bar}')

# ── 2. RTG vs Action vs State embedding magnitude ──
print(f'\n{"="*70}')
print('2. TOKEN TYPE IMPORTANCE (embedding weight norms)')
print('='*70)

rtg_embed = actor.embed_rtg.weight.data.cpu().numpy()     # [512, 2]
act_embed = actor.embed_action.weight.data.cpu().numpy()  # [512, 1]
state_norm_total = np.linalg.norm(state_embed)
rtg_norm_total = np.linalg.norm(rtg_embed)
act_norm_total = np.linalg.norm(act_embed)

print(f'  State embedding  total norm: {state_norm_total:.2f}')
print(f'  RTG embedding    total norm: {rtg_norm_total:.2f}')
print(f'  Action embedding total norm: {act_norm_total:.2f}')

# RTG: V vs C importance
rtg_v_norm = np.linalg.norm(rtg_embed[:, 0])
rtg_c_norm = np.linalg.norm(rtg_embed[:, 1])
print(f'\n  RTG-V (reward) weight norm: {rtg_v_norm:.2f}')
print(f'  RTG-C (cost)   weight norm: {rtg_c_norm:.2f}')
print(f'  V/C importance ratio: {rtg_v_norm / rtg_c_norm:.2f}')

# ── 3. Prediction Head ──
print(f'\n{"="*70}')
print('3. PREDICTION HEAD: which hidden dims drive action?')
print('='*70)

pred_head = actor.predict_action.weight.data.cpu().numpy().flatten()  # [512]
top_k = 10
top_idx = np.argsort(-np.abs(pred_head))[:top_k]
print(f'  Top {top_k} hidden dims (by |weight|):')
for idx in top_idx:
    print(f'    dim {idx:4d}: weight={pred_head[idx]:+.4f}')

# ── 4. Per-layer analysis ──
print(f'\n{"="*70}')
print('4. PER-LAYER ATTENTION: Q,K,V matrix norms')
print('='*70)

for i, block in enumerate(actor.transformer):
    q_norm = torch.norm(block.attn.query.weight).item()
    k_norm = torch.norm(block.attn.key.weight).item()
    v_norm = torch.norm(block.attn.value.weight).item()
    proj_norm = torch.norm(block.attn.proj.weight).item()
    print(f'  Layer {i}: Q={q_norm:.1f}  K={k_norm:.1f}  V={v_norm:.1f}  Proj={proj_norm:.1f} | Q/K ratio={q_norm/k_norm:.2f}')

# ── 5. Embedding correlation ──
print(f'\n{"="*70}')
print('5. STATE DIMENSION CORRELATION IN EMBEDDING SPACE')
print('='*70)

# Cosine similarity between state dimension embeddings
corr_matrix = np.zeros((16, 16))
for i in range(16):
    for j in range(16):
        corr_matrix[i, j] = np.dot(state_embed[:, i], state_embed[:, j]) / (
            np.linalg.norm(state_embed[:, i]) * np.linalg.norm(state_embed[:, j]) + 1e-10)

high_corr = []
for i in range(16):
    for j in range(i+1, 16):
        if abs(corr_matrix[i, j]) > 0.5:
            high_corr.append((state_names[i], state_names[j], corr_matrix[i, j]))

print('  Highly correlated state dims (|cos|>0.5):')
for n1, n2, c in sorted(high_corr, key=lambda x: -abs(x[2]))[:15]:
    sign = 'pos' if c > 0 else 'neg'
    print(f'    {n1:<15} <-> {n2:<15}  {c:+.3f} ({sign})')

# ── Save ──
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# State importance
ax = axes[0]
colors = ['#2196F3'] * 16
ax.barh(range(16), state_col_norms[np.argsort(state_col_norms)], color=colors)
ax.set_yticks(range(16))
ax.set_yticklabels([state_names[i] for i in np.argsort(state_col_norms)], fontsize=8)
ax.set_xlabel('L2 Norm of Embedding Weight')
ax.set_title('State Dimension Importance')

# V vs C
ax = axes[1]
ax.bar(['RTG-V (reward)', 'RTG-C (cost)'], [rtg_v_norm, rtg_c_norm], color=['#4CAF50', '#FF9800'])
ax.set_ylabel('Weight Norm')
ax.set_title('RTG Component Importance')

# Layer Q/K ratio
ax = axes[2]
qk_ratios = []
for block in actor.transformer:
    qk_ratios.append(torch.norm(block.attn.query.weight).item() /
                     torch.norm(block.attn.key.weight).item())
ax.plot(range(len(qk_ratios)), qk_ratios, 'o-', color='#E91E63')
ax.set_xlabel('Layer')
ax.set_ylabel('Q/K Norm Ratio')
ax.set_title('Query/Key Ratio per Layer')
ax.axhline(y=1.0, color='gray', linestyle='--')

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'weight_analysis.png'), dpi=150)
print(f'\nSaved: weight_analysis.png')

print('\nDone.')
