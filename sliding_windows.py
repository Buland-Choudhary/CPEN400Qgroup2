import pennylane as qml
import numpy as np
import os
import matplotlib.pyplot as plt
import numpy as np

# need to install these:
# pip install qiskit
# pip install pennylane-qiskit
# pip install pennylane
from qiskit import QuantumCircuit
from qiskit.circuit.library import RealAmplitudes
from qiskit.circuit import ParameterVector

num_qubits = 7
total_qubits = num_qubits + 2

dev = qml.device("default.qubit", wires=total_qubits)

def amp_encode(data):
    qml.AmplitudeEmbedding(features=data, wires=range(num_qubits), normalize=True)

# comment out the other not using ansatz. 
def ansatz(params, wires):
    # 1. PauliTwoDesign Pairwise CZ
    qml.StronglyEntanglingLayers(params, wires=wires)

    # # 2. RealAmplitudes Circular
    # ansatz2 = RealAmplitudes(num_qubits=num_qubits, reps=params.shape[0], entanglement="circular")
    # qc = QuantumCircuit(num_qubits)
    # qc.compose(ansatz2, inplace=True)
    # pl_qfunc = qml.from_qiskit(qc)


    # # 3. RealAmplitudes Full
    # ansatz3 = RealAmplitudes(num_qubits=num_qubits, reps=params.shape[0], entanglement="full")
    # qc = QuantumCircuit(num_qubits)
    # qc.compose(ansatz3, inplace=True)
    # pl_qfunc = qml.from_qiskit(qc)


    # # 4. RealAmplitudes Linear
    # ansatz4 = RealAmplitudes(num_qubits=num_qubits, reps=params.shape[0], entanglement="linear")
    # qc = QuantumCircuit(num_qubits)
    # qc.compose(ansatz4, inplace=True)
    # pl_qfunc = qml.from_qiskit(qc)

    # # 5. RealAmplitudes SCA
    # ansatz5 = RealAmplitudes(num_qubits=num_qubits, reps=params.shape[0], entanglement="sca")
    # qc = QuantumCircuit(num_qubits)
    # qc.compose(ansatz5, inplace=True)
    # pl_qfunc = qml.from_qiskit(qc)

@qml.qnode(dev, interface="autograd", diff_method="backprop")
def vqae_trainable(params, input_data):
    amp_encode(input_data)
    ansatz(params, wires=range(num_qubits))

    swap_wire1 = int(num_qubits - 1)
    swap_wire2 = int(num_qubits)
    hadamard_wire = int(num_qubits + 1)
    cswap_control = int(num_qubits + 1)
    cswap_target1 = int(num_qubits)
    cswap_target2 = int(num_qubits - 2)

    qml.SWAP(wires=[swap_wire1, swap_wire2])
    qml.Hadamard(wires=hadamard_wire)
    qml.CSWAP(wires=[cswap_control, cswap_target1, cswap_target2])
    qml.Hadamard(wires=hadamard_wire)

    return qml.probs(wires=hadamard_wire)

def swap_loss(params, input_data):
    probs = vqae_trainable(params, input_data)
    return probs[1]

filepath = r"<PLACE YOUR ABSOLUTE FILE PATH>"
with open(filepath, 'r') as f:
    data = [float(line.strip()) for line in f]
time_series = np.array(data)

window_size = 2**num_qubits

X = 2700 # this is the training data cut off  pull this from the data file name
train_indices = list(range(1, X)) + [23000, 52600, 52800] # our known anomolies pull this from the data file name
train_windows = [time_series[i:i + window_size] for i in train_indices if i + window_size <= len(time_series)]

num_layers = 2
params_shape = qml.StronglyEntanglingLayers.shape(num_layers, num_qubits)
params = np.random.uniform(0, 1, params_shape)

opt = qml.GradientDescentOptimizer(stepsize=0.1)
steps = 20

# save trained modle
params_file = "trained_params.npy"

def train_model(opt, steps, train_windows, params, params_file):
    for step in range(steps):
        print(f"Training Step {step+1}/{steps}")
        total_loss = 0
        for idx, window in enumerate(train_windows):
            params = opt.step(lambda p: swap_loss(p, window), params)
            loss = swap_loss(params, window)
            total_loss += loss
        print(f"Step {step+1} Completed. Average Loss: {total_loss / len(train_windows)}\n")

    np.save(params_file, params)
    print(f"Trained parameters saved to {params_file}")
    return params

if False:
    try:
        params = np.load(params_file)
        print(f"Loaded trained parameters from {params_file}")
    except Exception as e:
        print(f"Error loading parameters: {e}. Retraining...")
        params = train_model(opt, steps, train_windows, params, params_file)
else:
    params = train_model(opt, steps, train_windows, params, params_file)

test_indices = [500, 1000, 2000, 5000, 10000, 15000, 20000, 23000, 25000, 30000, 35000, 40000, 45000, 50000, 52600, 52800, 53000,53500, 54000, 54500] # test indicies for where we wanna see anomoly detection (can cluster around the anomoly)
for idx in test_indices:
    test_data = time_series[idx:idx + window_size]
    if len(test_data) == window_size:
        swap_test_result = vqae_trainable(params, test_data)[1]
        print(f"Test Swap-Test Probability (Anomaly Score) at index {idx}: {swap_test_result}")
    else:
        print(f"Window at index {idx} is too short.")
