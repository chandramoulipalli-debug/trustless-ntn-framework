"""Generate Fig8 (threshold sensitivity) and Fig9 (Paillier comparison)."""
import json, os, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams.update({
    'font.family': 'serif', 'font.size': 10,
    'axes.labelsize': 11, 'axes.titlesize': 11,
    'legend.fontsize': 9, 'xtick.labelsize': 9, 'ytick.labelsize': 9,
    'figure.dpi': 150, 'savefig.dpi': 300,
    'savefig.bbox': 'tight', 'lines.linewidth': 1.8, 'axes.grid': True,
    'grid.alpha': 0.3,
})

os.makedirs('../results/figures', exist_ok=True)

# ── Fig 8: Threshold Sensitivity ───────────────────────────────────────────
with open('../results/data/exp7_threshold_sensitivity.json') as f:
    d7 = json.load(f)

thetas = [r['theta']          for r in d7['theta_sweep']]
f1s    = [r['f1_mean']        for r in d7['theta_sweep']]
f1ci   = [r['f1_ci95']        for r in d7['theta_sweep']]
fprs   = [r['fpr_mean']       for r in d7['theta_sweep']]
precs  = [r['precision_mean'] for r in d7['theta_sweep']]
recs   = [r['recall_mean']    for r in d7['theta_sweep']]

# Analytic optimal values
analytic = d7['analytic_steady_state']
theta_opt_global = 0.20  # min T*_legit = 0.264 (UAV) > 0.20; max T*_mal = 0.062 (GEO) < 0.20

fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.2))

# Panel (a): F1 and FPR vs theta
ax1 = axes[0]
ax2 = ax1.twinx()
l1 = ax1.errorbar(thetas, f1s, yerr=f1ci, fmt='o-', color='#1f77b4',
                   capsize=4, label='F1 Score (left)')
l2 = ax2.plot(thetas, fprs, 's--', color='#d62728', label='FPR (right)')
ax1.axvline(theta_opt_global, color='green', linestyle=':', linewidth=1.5,
            label=f'theta*=0.20 (analytical)')
ax1.set_xlabel('Detection Threshold theta')
ax1.set_ylabel('F1 Score', color='#1f77b4')
ax2.set_ylabel('False Positive Rate', color='#d62728')
ax1.tick_params(axis='y', labelcolor='#1f77b4')
ax2.tick_params(axis='y', labelcolor='#d62728')
ax1.set_title('(a) F1 and FPR vs Threshold')
lines = [l1, l2[0], plt.Line2D([0],[0], color='green', linestyle=':',
                                 label='theta*=0.20')]
ax1.legend(lines, [l.get_label() for l in lines], fontsize=8, loc='center right')

# Panel (b): Precision and Recall vs theta
axes[1].plot(thetas, precs, '^-', color='#2ca02c', label='Precision')
axes[1].plot(thetas, recs,  'v--', color='#ff7f0e', label='Recall')
axes[1].axvline(theta_opt_global, color='green', linestyle=':', linewidth=1.5,
                label='theta*=0.20')
axes[1].set_xlabel('Detection Threshold theta')
axes[1].set_ylabel('Score')
axes[1].set_title('(b) Precision and Recall vs Threshold')
axes[1].legend(fontsize=8)

# Annotate key point
idx_opt = thetas.index(0.20)
axes[0].annotate(f'FPR=0\nF1={f1s[idx_opt]:.3f}',
                  xy=(0.20, f1s[idx_opt]), xytext=(0.25, 0.20),
                  arrowprops=dict(arrowstyle='->', color='black'),
                  fontsize=8, color='green')

plt.tight_layout()
plt.savefig('../results/figures/fig8_threshold_sensitivity.pdf')
plt.savefig('../results/figures/fig8_threshold_sensitivity.png')
plt.close()
print('Fig8 saved.')

# ── Fig 9: Paillier 1024 vs 2048-bit ──────────────────────────────────────
with open('../results/data/exp8_paillier_2048.json') as f:
    d8 = json.load(f)

nc   = [r['n_contributors']  for r in d8['key_bits_1024']]
t_1024 = [r['total_time_ms'] for r in d8['key_bits_1024']]
t_2048 = [r['total_time_ms'] for r in d8['key_bits_2048']]
s_1024 = [r['ciphertext_size_kb'] for r in d8['key_bits_1024']]
s_2048 = [r['ciphertext_size_kb'] for r in d8['key_bits_2048']]

fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.2))

axes[0].plot(nc, t_1024, 'o-', color='#1f77b4', label='1024-bit')
axes[0].plot(nc, t_2048, 's-', color='#d62728', label='2048-bit (NIST-compliant)')
axes[0].axhline(60000, color='gray', linestyle='--', linewidth=1,
                label='60s round budget')
axes[0].set_xlabel('Number of Contributors')
axes[0].set_ylabel('Total Latency (ms)')
axes[0].set_title('(a) Paillier HE Total Latency')
axes[0].legend(fontsize=8)

axes[1].plot(nc, s_1024, 'o-', color='#1f77b4', label='1024-bit')
axes[1].plot(nc, s_2048, 's-', color='#d62728', label='2048-bit')
axes[1].set_xlabel('Number of Contributors')
axes[1].set_ylabel('Ciphertext Size (KB)')
axes[1].set_title('(b) Aggregate Ciphertext Size')
axes[1].legend(fontsize=8)

plt.tight_layout()
plt.savefig('../results/figures/fig9_paillier_comparison.pdf')
plt.savefig('../results/figures/fig9_paillier_comparison.png')
plt.close()
print('Fig9 saved.')
print('All new figures saved to results/figures/')
