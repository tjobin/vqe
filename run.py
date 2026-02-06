from vqe import get_vqe_results_v2
from state_and_hamiltonian import get_state_and_hamiltonian
from optimizer import get_optimizer
from estimator import get_estimator
from utils import get_circuit_depth
from optimizer import SPSAHistory

def run_vqe_simulation(
        atomic_symbol,
        state_type,
        bond_lengths,
        n_shots_list,
        n_iters_list,
        depolarizing_errors,
        active_orbitals,
        n_elec,
        optimizer_name='spsa',
        regularization=1e-8,
        filename='results.txt'
    ):

    """
    Runs a VQE simulation for a given state type (UCCSD or EfficientSU2) at different bond lengths,
    numbers of shots, numbers of iterations, and depolarizing error scalings. Saves the results in a .txt file
    in the out/results/ folder and returns a dictionary with the results.
        Args:
            - atomic_symbol: str, either 'H2' or 'LiH'
            - state_type: str, either 'UCCSD' or 'EfficientSU2'
            - bond_lengths: list, of len N_dist, contains the bond lengths
            - n_shots_list: list, of len N_shots, contains the numbers of shots
            - n_iters_list: list, of len N_iters, contains the numbers of optimization iterations
            - depolarizing_errors: list, of len N_errors, contains the scalings of the depolarizing error probabilities
            - active_orbitals: int, number of active orbitals (2 or 4)
            - n_elec: int, number of electrons (2 or 4)
            - optimizer_name: str, name of the optimizer to be used (only 'spsa' implemented)
            - regularization: float, regularization coefficient for the optimizer
            - filename: str, name of the .txt file to be saved in the out/results/ folder
    """

    results = {}

    default_p1 = 0.001  # Default single-qubit depolarizing error probability
    default_p2 = 0.02   # Default two-qubit depolarizing error probability

    with open('out/results/' + filename, 'w') as file:
        file.write(f"{'n_shots':<10} {'n_iters':<10} {'dep_error':<12} {'energy_fav':<10} {'depth':<10}\n")  # Write header with spacing

        for n_shots in n_shots_list:
            for n_iters in n_iters_list:
                for dep_error in depolarizing_errors:
                    p1_scaled = dep_error * default_p1
                    p2_scaled = dep_error * default_p2
                    key = (n_shots, n_iters, dep_error)
                    results[key] = {'bond_lengths': [], 'energies_per_iter': [], 'energies_fav': [], 'depths': []}
                    history_tracker = SPSAHistory()
                    callback = history_tracker.callback

                    optimizer = get_optimizer(optimizer_name, max_iter=n_iters, regularization=regularization, callback=callback)
                    estimator = get_estimator(nqubits=2*active_orbitals, estimator_name='noisy', n_shots=n_shots, p_err_1q=p1_scaled, p_err_2q=p2_scaled)

                    for bond_length in bond_lengths:
                        if atomic_symbol == 'H2':
                            geometry = f'H 0 0 0; H 0 0 {bond_length}'
                        elif atomic_symbol == 'LiH':
                            geometry = f'Li 0 0 0; H 0 0 {bond_length}'
                        else:
                            raise ValueError("Atomic symbol not supported; must be either 'H2' or 'LiH'")
                        
                        state, hamiltonian = get_state_and_hamiltonian(state_type=state_type, geometry=geometry, basis_set='sto-3g', active_orb=active_orbitals, n_elec=n_elec)

                        _, energies, energy_fav = get_vqe_results_v2(
                            state=state,
                            hamiltonian=hamiltonian,
                            optimizer=optimizer,
                            estimator=estimator,
                            history_tracker=history_tracker,
                            filename=f'noisy/{state_type}/n_elec={n_elec}/no={active_orbitals}/shots{n_shots}_iters{n_iters}_scale{dep_error}'
                        )

                        depth = get_circuit_depth(state, 'ibm')
                        n_varparams = state.num_parameters

                        print(f"{state_type} Completed: shots={n_shots}, iters={n_iters}, dep_error={dep_error}, bond_length={bond_length}, energy_fav={energy_fav:.6f}, depth={depth}, n_varparams={n_varparams}")

                        results[key]['bond_lengths'].append(bond_length)
                        results[key]['energies_per_iter'].append(energies)
                        results[key]['energies_fav'].append(energy_fav)
                        results[key]['depths'].append(depth)

                        # Write energy_fav and depth to the file with spacing
                        file.write(f"{n_shots:<10} {n_iters:<10} {dep_error:<12.6f} {energy_fav:<10.6f} {depth:<10}\n")

    return results