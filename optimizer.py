from qiskit_algorithms.optimizers import optimizer
from qiskit_algorithms.optimizers.spsa import SPSA
import qiskit_algorithms
import numpy as np


qiskit_algorithms.utils.algorithm_globals.random_seed = 0



def get_optimizer(
        optimizer : str = 'spsa',
        max_iter : int = 100,
        regularization : float = 1e-8,
        callback = None
):
        if optimizer == 'spsa':
                optimizer = SPSA(maxiter=max_iter, blocking=True, regularization=regularization, allowed_increase=0.0, callback=callback)
                return optimizer


class SPSAHistory:
    def __init__(self):
        self.params = []
        self.values = []

    def callback(self, n_evals, params, fval, step_size, accepted):
        # Save a copy of the parameters and the energy value
        self.params.append(np.copy(params))
        self.values.append(fval)

from qiskit.primitives import Estimator

def validate_spsa_convergence(ansatz, hamiltonian, history_tracker, best_params_raw, high_shots=10000, n_avg=50):
    """
    Distinguishes between statistical noise artifacts and true convergence.
    
    Args:
        ansatz: The QuantumCircuit used in VQE.
        hamiltonian: The operator (SparsePauliOp) for LiH.
        history_tracker: The SPSAHistory object containing the trace.
        best_params_raw: The 'x' attribute from the optimizer result (best observed).
        high_shots: Number of shots for the validation check (should be >> training shots).
        n_avg: Number of last iterations to average for the 'Favorite' candidate.
    """
    
    # 1. Calculate the "Favorite" candidate (Polyak-Ruppert Average)
    # The paper suggests averaging parameters from the end of the trace 
    # to beat the sampling noise floor.
    if len(history_tracker.params) >= n_avg:
        # Take the last n_avg parameters and compute the mean
        last_params = np.array(history_tracker.params[-n_avg:])

        theta_favorite = np.mean(last_params, axis=0)
    else:
        print(f"Warning: Not enough iterations ({len(history_tracker.params)}) to average {n_avg}. Using final params.")
        theta_favorite = best_params_raw

    theta_best = best_params_raw

    # 2. Re-evaluate both candidates with HIGH precision
    # We create a fresh estimator with high shots to reduce variance
    validator = Estimator(options={"shots": high_shots})
    
    # Prepare inputs for the estimator (two jobs: one for best, one for favorite)
    # Note: Adjust 'circuits' and 'observables' lists based on your Qiskit version (Pre-1.0 vs 1.0)


    job = validator.run(
        circuits=[ansatz, ansatz],
        observables=[hamiltonian, hamiltonian],
        parameter_values=[np.array(theta_best).tolist(), np.array(theta_favorite).tolist()]
    )
    result = job.result()
    energy_best_checked = result.values[0]
    energy_fav_checked = result.values[1]
    
    # # 3. Analysis
    # print(f"--- Validation Results (Shots: {high_shots}) ---")
    # print(f"1. Best Observed (Raw):     {energy_best_checked:.6f} Ha")
    # print(f"2. Favorite (Averaged):     {energy_fav_checked:.6f} Ha")
    # print(f"-------------------------------------------")
    
    # if energy_best_checked > energy_fav_checked:
    #     print("Diagnosis: The 'Best' value was likely a STATISTICAL ARTIFACT.")
    #     print("Reason: The averaged parameters (Favorite) perform better than the specific point")
    #     print("that SPSA claimed was the minimum. The 512-shot run likely just found a lucky")
    #     print("noise fluctuation.")
    # else:
    #     print("Diagnosis: The 'Best' value is ROBUST.")
    #     print("Reason: The specific minimum is genuinely lower than the surrounding average.")
    #     print("If this energy is higher than your 512-shot run, your 1024-shot run might be")
    #     print("trapped in a local minimum (freezing effect).")

    return energy_best_checked, energy_fav_checked