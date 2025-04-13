# CPEN400Qgroup2

training_and_saving.py: This is the file where we can train the QAE with five ansatz and store the trained parameters.

testing_and_graphing.py: This file displays the trained parameters in graph. 

AnomalyDatasets_and_TrainedParams: In this folder, there are original datasets and trained parameters. For example, 028_UCR_Anomaly_DISTORTEDInternalBleeding17_1600_3198_3309.txt is the original dataset and its trained parameters are stored in these files:
028_trained_params.npy (trained parameters for PauliTwoDesign)
028_trained_params_circular.npy(trained parameters for RealAmplitudes circular entanglement)
028_trained_params_full.npy(trained parameters for RealAmplitudes full entanglement)
028_trained_params_linear.npy(trained parameters for RealAmplitudes linear entanglement)
028_trained_params_sca.npy(trained parameters for RealAmplitudes sca entanglement) 

Each of the six datasets has 5 trained parameter files.

# Installation
pip install qiskit
pip install pennylane-qiskit
pip install pennylane
pip install scipy

# Dataset's name:
028_UCR_Anomaly_DISTORTEDInternalBleeding17_1600_3198_3309.txt

028: number of dataset
1600: training the model from 1 to 1600 data indices. 
3198_3309: anomalies locates from 3198 to 3309.

