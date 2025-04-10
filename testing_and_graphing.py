'''
This script measure the performance of a trained quantum model on a time series
dataset using a swap test and reconstruction error (MSE). It visualizes the 
results, highlighting anomalies detected by the model. The script assumes that 
the model has been trained and the parameters are saved in a file named 
"trained_params.npy".

IMPORTANT: THIS SCRIPT GENERATES 2 SEPERATE GRAPHS. YOU NEED TO CLOSE THE FIRST GRAPH TO SEE THE SECOND ONE.
'''

import numpy as np
import matplotlib.pyplot as plt
import pennylane as qml

# === Configurations ===
num_qubits = 7
total_qubits = num_qubits + 2  # For swap test ancilla + extra wires
shots = 5000
reps = 2  # Number of ansatz layers
USE_QISKIT_ANSATZ = True # True for RealAmplitudes, False for PauliTwoDesign

# === Load Trained Parameters ===
trained_params = np.load("trained_params.npy")

# === Pennylane Device ===
dev = qml.device("default.qubit", wires=total_qubits, shots=shots)

# === Embedding Function ===
def amp_encode(data):
    qml.AmplitudeEmbedding(features=data, wires=range(num_qubits), normalize=True)

# === Ansatz (PauliTwoDesign) ===
def apply_ansatz(params):
    if USE_QISKIT_ANSATZ:
        reps = 2
        real_amp = RealAmplitudes(num_qubits, reps=reps, entanglement=entanglement)
        qc = QuantumCircuit(num_qubits)
        qc.compose(real_amp, inplace=True)
        qfunc = qml.from_qiskit(qc)
        qfunc(params.flatten())
    else:
        qml.StronglyEntanglingLayers(params, wires=range(num_qubits))

# === QNode: Reconstruction ===
@qml.qnode(dev, interface="autograd", diff_method="parameter-shift")
def reconstruct(input_data):
    amp_encode(input_data)
    apply_ansatz(trained_params)
    return qml.probs(wires=range(num_qubits))

# === QNode: Swap Test ===
@qml.qnode(dev, interface="autograd", diff_method="parameter-shift")
def swap_test(input_data):
    amp_encode(input_data)
    apply_ansatz(trained_params)

    h_wire = num_qubits + 1
    t1 = num_qubits
    t2 = num_qubits - 2

    qml.Hadamard(wires=h_wire)
    qml.CSWAP(wires=[h_wire, t1, t2])
    qml.Hadamard(wires=h_wire)

    return qml.probs(wires=h_wire)

# === Load Time Series Data  ===
filepath = "054_UCR_Anomaly_DISTORTEDWalkingAceleration5_2700_5920_5979.txt"
data = np.loadtxt(filepath)
time_series = np.array(data)

# Define test windows — evaluate on post-training region
X = 2700
test_indices = list(range(X, len(time_series) - 128))
test_windows = [time_series[i:i + 128] for i in test_indices]

# === Evaluate Performance ===
mse_scores = []
swap_scores = []

for window in test_windows:
    probs = reconstruct(window)
    mse = np.mean((window - probs)**2)
    swap = swap_test(window)[1]

    mse_scores.append(mse)
    swap_scores.append(swap)

# === GRAPH THE RESULTS ===
def moving_average(data, window_size):
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')

# === Compute Moving Average ===
window_size = 100
swap_ma = moving_average(swap_scores, window_size)
swap_ma_indices = test_indices[window_size-1:]  # Adjust indices

# Define anomaly threshold
threshold = np.mean(swap_ma) + 2 * np.std(swap_ma)
anomalies = [score > threshold for score in swap_ma]

# === Plot Enhanced Swap-Test Chart ===
plt.figure(figsize=(12, 6))

plt.plot(swap_ma_indices, swap_scores[window_size-1:], label="Swap-Test Measurements", color='lightgray')
plt.plot(swap_ma_indices, swap_ma, label="Moving Averaged", color='black')

# Highlight anomalies
for i, (idx, is_anomaly) in enumerate(zip(swap_ma_indices, anomalies)):
    if is_anomaly:
        plt.plot(idx, swap_ma[i], 'r.')  # Red dot for anomaly

plt.xlabel("Time Step")
plt.ylabel("SWAP-Test Measurement")
plt.title("Swap-Test Anomaly Detection")
plt.legend(["Moving Averaged", "Swap-Test Measurements", "Valid Detection Range"])
plt.grid(True)
plt.tight_layout()
plt.show()

# === DO THE SAME FOR MSE ===
mse_ma = moving_average(mse_scores, window_size)
mse_ma_indices = test_indices[window_size - 1:]

# Define MSE anomaly threshold (optional - same logic as swap)
mse_threshold = np.mean(mse_ma) + 2 * np.std(mse_ma)
mse_anomalies = [score > mse_threshold for score in mse_ma]

# === Plot MSE in a new window ===
plt.figure(figsize=(12, 6))

plt.plot(mse_ma_indices, mse_scores[window_size-1:], label="Raw MSE", color='lightgray')
plt.plot(mse_ma_indices, mse_ma, label="Moving Averaged", color='black')

# Highlight anomalies
for i, (idx, is_anomaly) in enumerate(zip(mse_ma_indices, mse_anomalies)):
    if is_anomaly:
        plt.plot(idx, mse_ma[i], 'r.')

plt.xlabel("Time Step")
plt.ylabel("Reconstruction MSE")
plt.title("Reconstruction Error (Postprocessed)")
plt.legend(["Moving Averaged", "Raw MSE", "Anomalies"])
plt.grid(True)
plt.tight_layout()
plt.show()
