"""
Analyze DGAB model attention patterns.

Usage:
    python analyze_dgab_attention.py

Outputs attention heatmaps and per-head statistics to exp_data/.
"""

import sys, os, pickle
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

_PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_PROJECT_ROOT, 'strategy_train_env'))

# DGAB model lives in autobidding project
_autobidding_root = 'D:/research/Experiment/autobidding'
if os.path.isdir(_autobidding_root):
    sys.path.insert(0, _autobidding_root)

from bidding_train_env.baseline.dgab.model_po import DGAB, DGABRollout

# ── Config ──────────────────────────────────────────

CKPT_DIR = 'D:/research/Experiment/autobidding/saved_model/dgab_400k_sparse'
DEVICE = 'cpu'
OUTPUT_DIR = os.path.join(_PROJECT_ROOT, 'exp_data', 'dgab_analysis')
os.makedirs(OUTPUT_DIR, exist_ok=True)

BLOCK_CONFIG = {
    'n_ctx': 1024, 'n_embd': 512, 'n_layer': 8, 'n_head': 16,
    'n_inner': 1024, 'activation_function': 'relu', 'n_position': 1024,
    'resid_pdrop': 0.1, 'attn_pdrop': 0.1,
}

# ── Load model ──────────────────────────────────────

print('Loading model...')
with open(os.path.join(CKPT_DIR, 'normalize_dict.pkl'), 'rb') as f:
    nd = pickle.load(f)

model = DGAB(
    base_state_dim=16, act_dim=1,
    hidden_size=512, max_ep_len=96, time_dim=8,
    block_config=BLOCK_CONFIG, actor_type='stack', critic_type='sequence',
    device=DEVICE,
)
ckpt = os.path.join(CKPT_DIR, 'complete_train.pt')
model.load_state_dict(torch.load(ckpt, map_location=DEVICE))
model.to(DEVICE)
model.eval()
print(f'Model loaded. Layers: {len(model.actor.transformer)} blocks, {BLOCK_CONFIG["n_head"]} heads each')

# ── Collect attention maps ──────────────────────────

# Capture attention from each block
attention_maps = {i: [] for i in range(len(model.actor.transformer))}

def hook_factory(layer_idx):
    def hook(module, input, output):
        # CausalSelfAttention stores _attn_map after forward
        if hasattr(module, '_attn_map'):
            attention_maps[layer_idx].append(module._attn_map.detach().cpu())
    return hook

hooks = []
for i, block in enumerate(model.actor.transformer):
    h = block.attn.register_forward_hook(hook_factory(i))
    hooks.append(h)

# ── Run sample forward passes ────────────────────────

# Build sample inputs: vary RTG, state to see different attention patterns
print('Running forward passes...')

# Scenario 1: early tick (high budget, high RTG)
state_early = torch.randn(1, 10, 16) * 0.5  # 10-token sequence
rtg_early = torch.randn(1, 10, 2) * 0.5
actions_early = torch.randn(1, 10, 1) * 0.5
timesteps = torch.arange(10).unsqueeze(0)

with torch.no_grad():
    _ = model.actor(rtg_early, state_early, actions_early, timesteps)

print(f'Collected attention from {len(attention_maps[0])} forward passes')
print(f'Each map shape: {attention_maps[0][0].shape}')  # [B, n_head, T, T]

# ── Analyze ─────────────────────────────────────────

# Average attention across all layers
avg_by_layer = {}
for i in range(len(model.actor.transformer)):
    if attention_maps[i]:
        # Average over batch dim, then over the single forward pass tokens
        attn = attention_maps[i][0]  # [1, n_head, 30, 30] (3 tokens × 10 steps)
        avg_by_layer[i] = attn

# Per-head attention entropy (higher = more distributed, lower = more focused)
print('\n--- Attention Entropy per Layer/Head ---')
entropy_data = []
for layer_idx, attn in avg_by_layer.items():
    n_heads = attn.shape[1]
    for h in range(n_heads):
        # Average attention over query positions (last dim)
        head_attn = attn[0, h]  # [30, 30]
        # Entropy of attention distribution per query position, then average
        eps = 1e-10
        entropy = -(head_attn * torch.log(head_attn + eps)).sum(dim=-1).mean().item()
        entropy_data.append({'layer': layer_idx, 'head': h, 'entropy': entropy})

df_entropy = pd.DataFrame(entropy_data)
print(df_entropy.groupby('layer')['entropy'].agg(['mean', 'std']).to_string())

# Most focused heads (lowest entropy)
most_focused = df_entropy.nsmallest(5, 'entropy')
print(f'\nMost focused heads: {most_focused.to_string(index=False)}')

# ── Token-type attention analysis ──
# The model interleaves [rtg, state, action] tokens (order: rtg, state, action)
# T=10 steps → 30 tokens: [R0,S0,A0, R1,S1,A1, ..., R9,S9,A9]
# Check: which token types does the model attend to?

T = 10
token_labels = []
for t in range(T):
    token_labels.extend([f'R{t}', f'S{t}', f'A{t}'])

# Build token-type attention matrix (3×3: from-type to to-type)
attn_last = attention_maps[len(model.actor.transformer)-1][0]  # last layer
avg_attn = attn_last[0].mean(dim=0)  # average over heads [30, 30]

type_attn = np.zeros((3, 3))  # from RTG/State/Action → to RTG/State/Action
for from_t in range(3):
    for to_t in range(3):
        from_idx = [i for i in range(30) if i % 3 == from_t]
        to_idx = [i for i in range(30) if i % 3 == to_t]
        type_attn[from_t, to_t] = avg_attn[to_idx][:, from_idx].mean().item()

print('\n--- Token-Type Attention (last layer, avg over heads) ---')
print('           To: RTG      To: State   To: Action')
type_names = ['From: RTG  ', 'From: State', 'From: Action']
for i, name in enumerate(type_names):
    print(f'  {name}  {type_attn[i,0]:.4f}      {type_attn[i,1]:.4f}      {type_attn[i,2]:.4f}')

# ── Temporal attention: does model look at recent or distant tokens? ──
temporal = np.zeros((T, T))
for from_t in range(T):
    for to_t in range(T):
        from_idx = [from_t*3, from_t*3+1, from_t*3+2]
        to_idx = [to_t*3, to_t*3+1, to_t*3+2]
        temporal[from_t, to_t] = avg_attn[to_idx][:, from_idx].mean().item()

print('\n--- Temporal Attention (query step → key step) ---')
print('  Higher values = more attention to that timestep')
print('  First few rows (early ticks):')
for t in range(min(5, T)):
    row = '  '.join(f'{temporal[t, j]:.3f}' for j in range(min(10, T)))
    print(f'  tick {t} → {row}')

# ── Save visualizations ─────────────────────────────

# 1. Temporal attention heatmap
fig, ax = plt.subplots(figsize=(8, 6))
im = ax.imshow(temporal, cmap='YlOrRd', aspect='auto')
ax.set_xlabel('Key (attended) tick'); ax.set_ylabel('Query (current) tick')
ax.set_title('DGAB Temporal Attention (last layer)')
plt.colorbar(im, ax=ax)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'temporal_attention.png'), dpi=150)
print(f'\nSaved: temporal_attention.png')

# 2. Token-type attention
fig, ax = plt.subplots(figsize=(5, 4))
im = ax.imshow(type_attn, cmap='YlOrRd', aspect='auto')
ax.set_xticks([0, 1, 2]); ax.set_xticklabels(['RTG', 'State', 'Action'])
ax.set_yticks([0, 1, 2]); ax.set_yticklabels(['RTG', 'State', 'Action'])
ax.set_xlabel('Key (attended)'); ax.set_ylabel('Query (current)')
ax.set_title('DGAB Token-Type Attention')
for i in range(3):
    for j in range(3):
        ax.text(j, i, f'{type_attn[i, j]:.3f}', ha='center', va='center')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'token_type_attention.png'), dpi=150)
print('Saved: token_type_attention.png')

# 3. Head entropy by layer
fig, ax = plt.subplots(figsize=(8, 4))
for layer in df_entropy['layer'].unique():
    sub = df_entropy[df_entropy['layer'] == layer]
    ax.scatter([layer] * len(sub), sub['entropy'], alpha=0.5, s=10)
ax.set_xlabel('Layer'); ax.set_ylabel('Attention Entropy')
ax.set_title('Per-Head Attention Entropy (lower = more focused)')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'head_entropy.png'), dpi=150)
print('Saved: head_entropy.png')

# ── Summary ─────────────────────────────────────────
print(f'\nDone. All outputs in {OUTPUT_DIR}/')
for h in hooks:
    h.remove()
