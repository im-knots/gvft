import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import signal
import seaborn as sns

# Load the data file
data = np.loadtxt('gvft_network_bio_prior_lamW_0.144_DF_0.019_t50_voltages.dat')

# Create column names based on the output file configuration
column_names = ['time', 'M1_v', 'M2L_v', 'M2R_v', 'M3L_v', 'M3R_v', 'M4_v', 'M5_v',
                'I1L_v', 'I1R_v', 'I2L_v', 'I2R_v', 'I3_v', 'I4_v', 'I5_v', 'I6_v', 'MI_v']

# Convert to DataFrame
df = pd.DataFrame(data, columns=column_names)

# 1. Basic statistics of the voltage data
print("Voltage Data Statistics (mV):")
print(df.iloc[:, 1:].describe())  # Skip time column

# 2. Check for any variations in the data
variation = df.iloc[:, 1:].max() - df.iloc[:, 1:].min()
print("\nMax variation in each neuron (mV):")
print(variation)

# 3. Create a heatmap of neuron activity over time
# Let's sample time points to make visualization manageable
time_samples = 100
sample_indices = np.linspace(0, len(df)-1, time_samples, dtype=int)
df_sampled = df.iloc[sample_indices]

# Normalize the data for better visualization since variations might be small
normalized_data = df_sampled.iloc[:, 1:].copy()
for col in normalized_data.columns:
    col_min = normalized_data[col].min()
    col_max = normalized_data[col].max()
    if col_max > col_min:
        normalized_data[col] = (normalized_data[col] - col_min) / (col_max - col_min)
    else:
        normalized_data[col] = 0  # Handle case where min=max (no variation)

# Create heatmap
plt.figure(figsize=(14, 8))
sns.heatmap(normalized_data.T, 
            xticklabels=[f"{t:.1f}" for t in df_sampled['time'].values[::10]],
            yticklabels=column_names[1:],
            cmap='viridis', 
            linewidths=0.5)
plt.title('Normalized Neural Activity Heatmap over Time')
plt.xlabel('Time (ms)')
plt.ylabel('Neuron')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('neuron_activity_heatmap.png', dpi=300)

# 4. Check if the input stimuli were properly applied
# Assuming I1L, I1R, I2L, I2R should receive input
plt.figure(figsize=(12, 6))
plt.plot(df['time'], df['I1L_v'], label='I1L_v')
plt.plot(df['time'], df['I1R_v'], label='I1R_v')
plt.plot(df['time'], df['I2L_v'], label='I2L_v')
plt.plot(df['time'], df['I2R_v'], label='I2R_v')
plt.title('Sensory Neurons Detail View (Should Receive Input Stimuli)')
plt.xlabel('Time (ms)')
plt.ylabel('Membrane Potential (mV)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('sensory_neurons_detail.png', dpi=300)

# 5. Visualize voltage histograms to check for multimodal distributions
plt.figure(figsize=(14, 10))
for i, col in enumerate(column_names[1:]):
    plt.subplot(4, 5, i+1)
    plt.hist(df[col], bins=50, alpha=0.7)
    plt.title(col)
    plt.xlabel('Voltage (mV)')
    if i % 5 == 0:
        plt.ylabel('Frequency')
plt.tight_layout()
plt.savefig('voltage_histograms.png', dpi=300)

# 6. Check temporal structure via autocorrelation
plt.figure(figsize=(14, 10))
max_lag = 1000  # Number of sample points for autocorrelation
for i, col in enumerate(column_names[1:]):
    plt.subplot(4, 5, i+1)
    signal_data = df[col].values
    # Mean-center the data
    signal_data = signal_data - np.mean(signal_data)
    if np.sum(signal_data**2) > 0:  # Check if signal has variance
        acorr = np.correlate(signal_data, signal_data, mode='full')
        acorr = acorr[len(acorr)//2:]  # Take positive lags
        acorr = acorr / acorr[0]  # Normalize
        plt.plot(acorr[:max_lag])
    else:
        plt.text(0.5, 0.5, "No variation", ha='center', va='center')
    plt.title(f'Autocorr: {col}')
plt.tight_layout()
plt.savefig('autocorrelation.png', dpi=300)

# 7. Look for any deviations from resting potential
deviations = np.abs(df.iloc[:, 1:] - df.iloc[:, 1:].mean())
max_deviations = deviations.max()
print("\nMaximum deviations from mean (mV):")
print(max_deviations)

plt.show()