from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, depolarizing_error
from qiskit_aer.primitives import EstimatorV2
from qiskit.primitives import StatevectorEstimator
import numpy as np


def get_estimator(nqubits, estimator_name, n_shots, p_err_1q = 0.001, p_err_2q = 0.02):

    if estimator_name == 'noiseless':
        estimator = StatevectorEstimator()

    elif estimator_name == 'noisy':

        # --- 1. SETUP NOISY ESTIMATOR V2 ---
        # Define Noise Model
        noise_model = NoiseModel(basis_gates=['id', 'rz', 'sx', 'x', 'cx'])

        # Add errors (Depolarizing noise on single and 2-qubit gates)
        noise_model.add_all_qubit_quantum_error(depolarizing_error(p_err_1q, 1), ["u1", "u2", "u3", "rz", "sx", "x"])
        noise_model.add_all_qubit_quantum_error(depolarizing_error(p_err_2q, 2), ["cx"])
        
        # Define Connectivity (Coupling Map)
        # Essential for valid noise simulation!
        if nqubits == 6:
            coupling_map = [
                [0, 1], [1, 0],
                [1, 2], [2, 1],
                [2, 3], [3, 2],
                [3, 4], [4, 3],
                [4, 5], [5, 4],
                [5, 0], [0, 5]  # Closes the ring
            ]
        else:
            coupling_map = [
                [0, 1], [1, 0],
                [1, 2], [2, 1],
                [2, 3], [3, 2]
            ]
        
        # Define your configuration dict
#         options = {
#             "backend_options": {
#                 "noise_model": noise_model,
#                 "coupling_map": coupling_map,  # Optional but recommended with noise
#                 "basis_gates": noise_model.basis_gates
#             },
#             "run_options": {
#                 "seed": 42,
#                 "shots": n_shots
#             }
# }
#         estimator = EstimatorV2(options=options)

        # 2. Instantiate the Simulator Backend FIRST (The "Solid Source")
        # We pass the noise model directly to the backend class.
        backend = AerSimulator(
            noise_model=noise_model,
            coupling_map=coupling_map,
            basis_gates=noise_model.basis_gates
        )

        # 3. Create Estimator from the Backend
        # This automatically configures the estimator to use the noisy backend.
        estimator = EstimatorV2.from_backend(backend)

        # 4. Set Shot Noise (The "default_shots" is correct, but we set it on the object)
        # This overrides the default_precision=0.0 and enables shot noise.
        estimator.options.default_precision =  0 # 1 / np.sqrt(n_shots)
        estimator.options.seed_simulator = 0
    return estimator