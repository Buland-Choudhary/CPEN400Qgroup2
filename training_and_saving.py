import numpy as np
import pennylane as qml
from qiskit import QuantumCircuit
from qiskit.circuit.library import RealAmplitudes
from scipy.optimize import minimize

num_qubits = 7
total_qubits = num_qubits + 2  
shots = 5000


dev = qml.device("default.qubit", wires=total_qubits, shots=shots)

filepath = "data_set_file.txt"
data = np.loadtxt(filepath)
time_series = np.array(data)

X = 2700  # Training cutoff can be differ based on dataset
train_indices = list(range(1, X))
train_windows = [time_series[i:i + 128] for i in train_indices if i + 128 <= len(time_series)]

USE_QISKIT_ANSATZ = False  # Can change this boolian for runninf whether RealAmplitudes or PauliTwoDesign
ANZ_ENTANGLE = ["circular", "full", "linear", "sca"] 

def amp_encode(data):
    qml.AmplitudeEmbedding(features=data, wires=range(num_qubits), normalize=True)

def apply_ansatz(params, entanglement=None):
    if USE_QISKIT_ANSATZ:
        reps = 2  
        real_amp = RealAmplitudes(num_qubits, reps=reps, entanglement=entanglement)
        qc = QuantumCircuit(num_qubits)
        qc.compose(real_amp, inplace=True)
        qfunc = qml.from_qiskit(qc)
        qfunc(params.flatten())  
    else:
        qml.StronglyEntanglingLayers(params, wires=range(num_qubits))

@qml.qnode(dev, interface="autograd", diff_method="parameter-shift")
def reconstruct(params, input_data, entanglement=None):
    if USE_QISKIT_ANSATZ:
        with qml.tape.QuantumTape() as tape:
            amp_encode(input_data)
    else:
        amp_encode(input_data)

    apply_ansatz(params, entanglement)
    return qml.probs(wires=range(num_qubits))


@qml.qnode(dev, interface="autograd", diff_method="parameter-shift")
def swap_test(params, input_data, entanglement=None):
    amp_encode(input_data)
    apply_ansatz(params, entanglement)

    h_wire = num_qubits + 1
    t1 = num_qubits
    t2 = num_qubits - 2

    qml.Hadamard(wires=h_wire)
    qml.CSWAP(wires=[h_wire, t1, t2])
    qml.Hadamard(wires=h_wire)

    return qml.probs(wires=h_wire)

# Calculate mean squared error (MSE)
def reconstruction_loss(params, input_data, entanglement=None):
    probs = reconstruct(params, input_data, entanglement)
    return np.mean((input_data - probs)**2)

# Anomaly score based on Swap-Test measurement
def swap_loss(params, input_data, entanglement=None):
    probs = swap_test(params, input_data, entanglement)
    return probs[1]

def train(params, train_windows, entanglement=None, maxiter=45):
    def cost_fn(flat_params):
        reshaped_params = flat_params.reshape(params.shape)
        return np.mean([swap_test(reshaped_params, window, entanglement) for window in train_windows])
    
    flat_params = params.flatten()
    result = minimize(cost_fn, flat_params, method='COBYLA', options={'maxiter': maxiter})
    return result.x.reshape(params.shape)


if USE_QISKIT_ANSATZ:
    reps = 2
    real_amp = RealAmplitudes(num_qubits, reps=reps, entanglement="full")
    num_params = real_amp.num_parameters
    params = np.random.uniform(0, 2*np.pi, size=(1, num_params))
else:
    reps = 2
    params_shape = qml.StronglyEntanglingLayers.shape(reps, num_qubits)
    params = np.random.uniform(0, 2*np.pi, size=params_shape)


params_file = "trained_params.npy"

try:
    params = np.load(params_file)
    print("Loaded trained parameters.")
except FileNotFoundError:
    print("Training the model...")
    if USE_QISKIT_ANSATZ:
        for entanglement in ANZ_ENTANGLE:
            params = np.random.uniform(0, 2*np.pi, size=(1, num_params)) 
            trained_params = train(params, train_windows, entanglement)
            np.save(f"trained_params_{entanglement}.npy", trained_params)
            print(f"Training complete for {entanglement}. Saved to file.")
    else:
        trained_params = train(params, train_windows)
        # Save Trained Parameters
        np.save(params_file, trained_params)
        print("Training complete. Saved to file.")