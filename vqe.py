from qiskit.quantum_info import Statevector, SparsePauliOp
from qiskit_algorithms.optimizers.spsa import SPSA
from qiskit_algorithms.minimum_eigensolvers.vqe import VQE
from qiskit.primitives import StatevectorEstimator
import numpy as np
from qiskit import transpile
from optimizer import validate_spsa_convergence
from qiskit_aer.primitives import EstimatorV2
from qiskit_algorithms.optimizers import SPSA


def get_vqe_results(
        state: Statevector,
        hamiltonian: SparsePauliOp,
        optimizer = SPSA,
        estimator = StatevectorEstimator,
        filename = 'default_filename'
    ):
    """
    Returns the VQE iterations and energies for a given state, hamiltonian,
    optimization method and estimator.
    """
    
    iters = []
    energies = []

    fout = open(f'out/{filename}.out','w')
    fout.write("QITE simulation \n")
    fout.write(f"{'Iteration':>10} {'E':>10} {'∆E':>10}\n")
    def my_vqe_callback(eval_count, parameters, mean, metadata):
        iters.append(eval_count)
        energies.append(mean)
        energy_change = energies[-1] - energies[-2] if len(energies) > 1 else 0
        fout.write(f"{eval_count:>10} {mean:>10.6f} {energy_change:>10.6f} \n")


    vqe = VQE(
            estimator=estimator,
            ansatz = state,
            optimizer=optimizer,
            callback=my_vqe_callback
        )
    
    _ = vqe.compute_minimum_eigenvalue(hamiltonian).eigenvalue.real
    fout.close()

    return iters, energies



# def get_vqe_results_v2(
#         state,              # Your quantum circuit (ansatz)
#         hamiltonian,        # SparsePauliOp
#         optimizer=SPSA(maxiter=100), 
#         estimator=EstimatorV2(),
#         history_tracker=None, # For convergence validation
#         filename='vqe_v2_log'
#     ):
    
    

#     # --- 2. TRANSPILE ANSATZ (ISA CIRCUIT) ---
#     # V2 requires circuits to be "ISA" (Instruction Set Architecture) compliant.
#     # We must transpile against the specific coupling map and basis gates of our noise model.
#     coupling_map = estimator._backend.coupling_map
#     target_basis = estimator._backend._basis_gates()

#     isa_ansatz = transpile(
#         state, 
#         coupling_map=coupling_map, 
#         basis_gates=target_basis, 
#         optimization_level=3,
#         seed_transpiler=0
#     )
    
#     # Map Hamiltonian to physical qubits if necessary (usually handled automatically by SparsePauliOp)
#     isa_hamiltonian = hamiltonian

#     # --- 3. OPTIMIZATION LOOP ---
#     iters = []
#     energies = []
    
#     # Prepare logging
#     fout = open(f'out/{filename}.out', 'w')
#     fout.write(f"VQE Simulation (EstimatorV2)\n")
#     fout.write(f"{'Iter':>10} {'Energy (Hartree)':>20}\n")
    
#     def cost_func(params):
#         # 1. Create a PUB (Primitive Unified Bloc)
#         # Format: (circuit, observables, parameter_values)
#         pub = (isa_ansatz, isa_hamiltonian, params)
        
#         # 2. Run Estimator
#         job = estimator.run([pub])
#         result = job.result()[0] # Result for the first PUB
        
#         # 3. Extract Energy
#         # EstimatorV2 returns expectation values in data.evs
#         current_energy = result.data.evs
        
#         # 4. Callback / Logging
#         iter_count = len(energies)
#         iters.append(iter_count)
#         energies.append(current_energy)
        
#         fout.write(f"{iter_count:>10} {current_energy:>20.6f}\n")
        
#         return current_energy

#     # Initialize Parameters
#     # If your ansatz has parameters, initialize them.
#     num_params = isa_ansatz.num_parameters
#     x0 = np.random.uniform(-np.pi, np.pi, num_params)

#     # Run Optimizer
#     result = optimizer.minimize(fun=cost_func, x0=x0)  # Hook in the tracker)

#     energy_best_checked, energy_fav_checked = validate_spsa_convergence(
#         ansatz=isa_ansatz,
#         hamiltonian=isa_hamiltonian,
#         history_tracker=history_tracker,
#         best_params_raw=result.x,
#         high_shots=10000, # Significantly higher than 512 or 1024
#         n_avg=50
#     )
#     fout.close()
    
#     return iters, energies, energy_fav_checked

def get_vqe_results_v2(
        state,              
        hamiltonian,        
        optimizer=SPSA(maxiter=100), 
        estimator=EstimatorV2(),
        history_tracker=None, 
        filename='vqe_v2_log'
    ):
    
    # ... [TRANSPILE CODE from your snippet remains here] ...
    coupling_map = estimator._backend.coupling_map
    target_basis = estimator._backend._basis_gates()
    isa_ansatz = transpile(state, coupling_map=coupling_map, basis_gates=target_basis, optimization_level=3, seed_transpiler=0)
    isa_hamiltonian = hamiltonian

    iters = []
    energies = []
    
    fout = open(f'out/{filename}.out', 'w')
    fout.write(f"VQE Simulation (EstimatorV2)\n")
    fout.write(f"{'Iter':>10} {'Energy (Hartree)':>20}\n")
    
    def cost_func(params):
        pub = (isa_ansatz, isa_hamiltonian, params)
        job = estimator.run([pub])
        result = job.result()[0] 
        current_energy = result.data.evs
        
        iter_count = len(energies)
        iters.append(iter_count)
        energies.append(current_energy)
        
        fout.write(f"{iter_count:>10} {current_energy:>20.6f}\n")
        return current_energy

    # --- Run Optimizer ---
    num_params = isa_ansatz.num_parameters
    x0 = np.random.uniform(-np.pi, np.pi, num_params)
    result = optimizer.minimize(fun=cost_func, x0=x0) 

    # --- NEW: Calculate Quantum Variance ---
    # We use the optimal parameters (result.x) found by SPSA.
    # Note: EstimatorV2 with Aer often provides variance in metadata. 
    # If not, we estimate it from standard error (stds) assuming shots are known.
    
    pub_final = (isa_ansatz, isa_hamiltonian, result.x)
    job_final = estimator.run([pub_final])
    res_final = job_final.result()[0]
    
    # Attempt to extract exact variance from metadata (available in Aer)
    # If using StatevectorEstimator (exact), this might be 0.0 or exact variance.
    try:
        # Check standard metadata location for 'variance'
        q_variance = res_final.metadata.get('variance', 0.0)
    except AttributeError:
        # Fallback for EstimatorV2 if metadata is missing/different structure
        # Var approx = (Standard Error)^2 * shots
        # We assume standard error is in data.stds
        q_variance = (res_final.data.stds)**2 * estimator.options.default_shots 

    # Run your validation check (unchanged)
    energy_best_checked, energy_fav_checked = validate_spsa_convergence(
        ansatz=isa_ansatz,
        hamiltonian=isa_hamiltonian,
        history_tracker=history_tracker,
        best_params_raw=result.x,
        high_shots=10000, 
        n_avg=50
    )
    fout.close()
    
    # Return variance as the 4th value
    return iters, energies, energy_fav_checked, q_variance