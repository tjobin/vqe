from state_and_hamiltonian import get_fci_energy
from _plots import (
                    make_convergence_plot,
                    make_convergence_plots_per_error,
                    make_pes_plots_per_nshots,
                    make_pes_per_niters,
                    make_convergence_plots_per_shots)
import numpy as np
from run import run_vqe_simulation


# Molecular system parameters
state_types = ['UCCSD', 'EfficientSU2']

charge = 0
spin = 0
active_orb = 3
n_elec = 4

# Optimizer parameters
op_name = 'spsa' # or 

# Estimator parameter
noise = 'noisy'

distances = [1.595,] # np.arange(1.0, 4.0, 0.1)
nshots_list = [1,] #, 512, 1024, 2048]
niters_list = [1,] #, 200, 300]
error_scalings= [1,]#[0, 1e-2, 5e-2, 1e-1, 5e-1, 1]



results_uccsd = run_vqe_simulation(
        state_type='UCCSD',
        bond_lengths = distances,
        n_shots_list= nshots_list,
        n_iters_list= niters_list,
        depolarizing_errors=error_scalings,
        active_orbitals=active_orb,
        n_elec=n_elec,
        optimizer_name=op_name,
        filename = f'LiH_UCCSD_results_niters{niters_list[0]}_error{error_scalings[0]}.txt'
)
results_hea = run_vqe_simulation(
        state_type='EfficientSU2',
        bond_lengths = distances,
        n_shots_list= nshots_list,
        n_iters_list= niters_list,
        depolarizing_errors=error_scalings,
        active_orbitals=active_orb,
        n_elec=n_elec,
        optimizer_name=op_name,
        filename = f'LiH_HEA_results_niters{niters_list[0]}_error{error_scalings[0]}.txt'
)


energies_per_type_per_shots = [[] for _ in range(len(nshots_list))]
energies_per_type_per_error = [[] for _ in range(len(error_scalings))]

iters = np.arange(niters_list[0])
fci_energy = get_fci_energy(active_orb=active_orb, n_elec=n_elec)



# for i, n_shots in enumerate(nshots_list):
#         key_uccsd = (n_shots, niters_list[0], error_scalings[0])
#         key_hea = (n_shots, niters_list[0], error_scalings[0])

#         energies_uccsd = results_uccsd[key_uccsd]['energies_per_iter'][0]
#         energies_hea = results_hea[key_hea]['energies_per_iter'][0]

#         energies_per_type_per_shots[i] = [energies_uccsd, energies_hea]
for i, error_scaling in enumerate(error_scalings):
        key_uccsd = (nshots_list[0], niters_list[0], error_scaling)
        key_hea = (nshots_list[0], niters_list[0], error_scaling)

        # energies_uccsd = results_uccsd[key_uccsd]['energies_per_iter'][0]
        energies_hea = results_hea[key_hea]['energies_per_iter'][0]

        energies_per_type_per_error[i] = [energies_hea] # [energies_uccsd, energies_hea]

make_convergence_plot(iters, energies_per_type_per_error[0], fci_energy, filename=f'LiH_HEA_convergence_plot_niters{niters_list[0]}_error{error_scalings[0]}')
# make_convergence_plots_per_error(iters, energies_per_type_per_error, error_scalings, fci_energy, filename=f'LiH_convergence_plot_per_error_bisbis')
# make_convergence_plots_per_shots(iters, energies_per_type_per_shots, nshots_list, fci_energy, filename=f'LiH_convergence_plot_per_shots_scale={error_scalings[0]}')




