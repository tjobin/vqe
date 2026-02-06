from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, depolarizing_error
from qiskit_aer.primitives import EstimatorV2
from qiskit.primitives import StatevectorEstimator


def get_estimator(
        nqubits: int = 6,
        estimator_name: str = 'noisy',
        n_shots: int = 0, # Not useful in this project, see note about it below
        p_err_1q: float = 0.001,
        p_err_2q: float = 0.02
        ):

    """
    Returns a qiskit Estimator instance
        Args:
            - nqubits: int, number of qubits of the system
            - estimator_name: str, name of the type of qiskit Estimator; either "noiseless" for the exact 
                StatevectorEstimator or "noisy" for the noisy AerEstimator
            - p_err_1q: float, single-qubit depolarizing error probability
            - p_err_2q: float, two-qubit depolarizing error probability
        Returns:
            - estimator: qiskit.primitives.EstimatorV2 or qiskit_aer.primitives.StatevectorEstimator instance
    """

    if estimator_name == 'noiseless':
        estimator = StatevectorEstimator()

    elif estimator_name == 'noisy':

        # Define Noise Model
        noise_model = NoiseModel(basis_gates=['id', 'rz', 'sx', 'x', 'cx'])

        # Add errors (Depolarizing noise on single and 2-qubit gates)
        noise_model.add_all_qubit_quantum_error(depolarizing_error(p_err_1q, 1), ["u1", "u2", "u3", "rz", "sx", "x"])
        noise_model.add_all_qubit_quantum_error(depolarizing_error(p_err_2q, 2), ["cx"])
        
        # Defines the qubit connectivity (here, Heavy-Hex) for 4 or 6 qubits
        if nqubits == 6:
            coupling_map = [
                [0, 1], [1, 0],
                [1, 2], [2, 1],
                [2, 3], [3, 2],
                [3, 4], [4, 3],
                [4, 5], [5, 4],
                [5, 0], [0, 5]
            ]
        elif nqubits ==4:
            coupling_map = [
                [0, 1], [1, 0],
                [1, 2], [2, 1],
                [2, 3], [3, 2],
                [2, 0], [0, 2]
            ]
        else:
            raise ValueError("Number of qubits not supported; must be either 4 or 6")
       
        # Instantiate the simulator backend
        # We pass the noise model directly to the backend class.
        backend = AerSimulator(
            noise_model=noise_model,
            coupling_map=coupling_map,
            basis_gates=noise_model.basis_gates
        )

        # Create Estimator from the Backend
        # This automatically configures the estimator to use the noisy backend.
        estimator = EstimatorV2.from_backend(backend)

        # Set shot noise via default_precision setting
        # Note : here, shot noise is essentially set to 0 because the AerEstimator instance adds it
        # as posterior Gaussian noise, which is ansatz-independent.
        if n_shots > 0:
            estimator.options.default_precision = 1 / (n_shots ** 0.5)  # Standard deviation of the Gaussian noise
        else:
            estimator.options.default_precision =  0
        estimator.options.seed_simulator = 0
    return estimator