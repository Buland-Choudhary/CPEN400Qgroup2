import matplotlib.pyplot as plt
import numpy as np

test_results = []
valid_indices = []

for idx in test_indices:
    test_data = time_series[idx:idx + window_size]
    if len(test_data) == window_size:
        swap_test_result = vqae_trainable(params, test_data)[1]
        test_results.append(swap_test_result)
        valid_indices.append(idx)

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(18, 10), sharex=True, 
                               gridspec_kw={'height_ratios': [2, 1]})

ax1.plot(time_series, label='Time Series', color='#1f77b4', alpha=0.8, linewidth=1)
ax1.set_ylabel('Value', fontsize=12)

# Hghlight anomalous iregions based on threshold
threshold = 0.5  
anomaly_mask = np.array(test_results) > threshold
anomaly_indices = np.array(valid_indices)[anomaly_mask]

for idx in anomaly_indices:
    ax1.axvspan(idx, idx+window_size, color='red', alpha=0.2)

# Mark anomalies with vertical lines
known_anomalies = [23000, 52600, 52800]
for anomaly in known_anomalies:
    ax1.axvline(x=anomaly, color='green', linestyle='--', linewidth=1.5, 
                alpha=0.7, label='Known Anomaly' if anomaly == known_anomalies[0] else "")

# Plot 2
ax2.scatter(valid_indices, test_results, color='red', s=60, 
           edgecolors='black', linewidths=0.5, label='Anomaly Score', zorder=3)
ax2.plot(valid_indices, test_results, color='red', alpha=0.4, linewidth=1, zorder=2)

ax2.set_xlabel('Time Index', fontsize=12)
ax2.set_ylabel('Anomaly Score', fontsize=12)
ax2.set_ylim(0, 1.1)  # Give some headroom for markers

ax1.grid(True, linestyle=':', alpha=0.6)
ax2.grid(True, linestyle=':', alpha=0.6)

# titles and legends
ax1.set_title('Time Series with Anomaly Detection', fontsize=14, pad=20)
ax2.set_title('Anomaly Scores', fontsize=12, pad=15)

handles1, labels1 = ax1.get_legend_handles_labels()
handles2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(handles1 + handles2, labels1 + labels2, loc='upper right', 
          bbox_to_anchor=(1, 1.15), ncol=3)

plt.tight_layout()
plt.subplots_adjust(hspace=0.1)

if len(anomaly_indices) > 0:
    for i, idx in enumerate(known_anomalies[:2]): 
        axins = ax1.inset_axes([0.1 + i*0.4, 0.6, 0.25, 0.3])
        axins.plot(time_series[max(0,idx-500):idx+500], color='#1f77b4')
        axins.axvline(x=500, color='green', linestyle='--', alpha=0.7)
        axins.set_title(f'Zoom: Index ~{idx}')
        ax1.indicate_inset_zoom(axins, edgecolor="black")

plt.show()
