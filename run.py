from vqe import get_vqe_results_v2
from state_and_hamiltonian import get_state_and_hamiltonian
from optimizer import get_optimizer
from estimator import get_estimator
from utils import get_circuit_depth
from optimizer import SPSAHistory

def run_vqe_simulation(
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
    results = {}

    default_p1 = 0.001  # Default depolarizing error probability
    default_p2 = 0.02   # Default two-qubit depolarizing error probability

    with open('out/results/' + filename, 'w') as file:
        file.write(f"{'n_shots':<10} {'n_iters':<10} {'dep_error':<12} {'energy_fav':<10} {'depth':<10} {'qvariance':<10} \n")  # Write header with spacing

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
                        geometry = f'Li 0 0 0; H 0 0 {bond_length}'
                        state, hamiltonian = get_state_and_hamiltonian(state_type=state_type, geometry=geometry, basis_set='sto-3g', active_orb=active_orbitals, n_elec=n_elec)

                        print(hamiltonian.size)
                        _, energies, energy_fav, qvariance = get_vqe_results_v2(
                            state=state,
                            hamiltonian=hamiltonian,
                            optimizer=optimizer,
                            estimator=estimator,
                            history_tracker=history_tracker,
                            filename=f'noisy/{state_type}/n_elec={n_elec}/no={active_orbitals}/shots{n_shots}_iters{n_iters}_scale{dep_error}'
                        )

                        depth = get_circuit_depth(state, 'ibm')
                        n_varparams = state.num_parameters

                        print(f"{state_type} Completed: shots={n_shots}, iters={n_iters}, dep_error={dep_error}, bond_length={bond_length}, energy_fav={energy_fav:.6f}, qvariance = {qvariance}, depth={depth}, n_varparams={n_varparams}")

                        results[key]['bond_lengths'].append(bond_length)
                        results[key]['energies_per_iter'].append(energies)
                        results[key]['energies_fav'].append(energy_fav)
                        results[key]['depths'].append(depth)

                        # Write energy_fav and depth to the file with spacing
                        file.write(f"{n_shots:<10} {n_iters:<10} {dep_error:<12.6f} {energy_fav:<10.6f} {depth:<10} {qvariance:<10.6f}\n")

    return results