from state_and_hamiltonian import get_fci_energy
from _plots import make_convergence_plots_per_param
import numpy as np
from run import run_vqe_simulation

### Main script to run the VQE simulations for H2 or LiH and generate the convergence plots
### for different error scalings and numbers of shots. 
### The results are saved in the out/results/ folder and the plots in the figs/ folder.
### Example below : Plot the convergence curves of ground-state energy for LiH at 1.595 Å with UCCSD and EfficientSU2 ansatze,
### in the noiseless case and with a depolarizing error scaling of 1, with 100 iterations and no shot noise (nshots=0).


# Molecular system parameters
atomic_symbol = 'LiH'
state_types = ['UCCSD', 'EfficientSU2']

charge = 0
spin = 0
active_orb = 2
n_elec = 2

# Optimizer parameters
op_name = 'spsa'

# Estimator parameter
noise = 'noisy'

# VQE simulation parameters
distances = [1.595,]
nshots_list = [0,]
niters_list = [100]
error_scalings= [0, 1]


# Compute results for each type of ansatz and each error scaling, and save them in .txt files
# in the out/results/ folder
results_uccsd = run_vqe_simulation(
        atomic_symbol=atomic_symbol,
        state_type='UCCSD',
        bond_lengths = distances,
        n_shots_list= nshots_list,
        n_iters_list= niters_list,
        depolarizing_errors=error_scalings,
        active_orbitals=active_orb,
        n_elec=n_elec,
        optimizer_name=op_name,
        filename = f'LiH_UCCSD_results_noiseless_vs_noisy.txt'
)
results_hea = run_vqe_simulation(
        atomic_symbol=atomic_symbol,
        state_type='EfficientSU2',
        bond_lengths = distances,
        n_shots_list= nshots_list,
        n_iters_list= niters_list,
        depolarizing_errors=error_scalings,
        active_orbitals=active_orb,
        n_elec=n_elec,
        optimizer_name=op_name,
        filename = f'LiH_HEA_results_noiseless_vs_noisy.txt'
)

# create list to store the energies for each type of ansatz and each error scaling
energies_per_type_per_error = [[] for _ in range(len(error_scalings))] 

# compute reference FCI energy for the system in the active space
fci_energy = get_fci_energy(active_orb=active_orb, n_elec=n_elec)

# extract the energies per iteration for each type of ansatz and each error scaling
# from the results dictionaries
for i, error_scaling in enumerate(error_scalings):
        key_uccsd = (nshots_list[0], niters_list[0], error_scaling)
        key_hea = (nshots_list[0], niters_list[0], error_scaling)

        energies_uccsd = results_uccsd[key_uccsd]['energies_per_iter'][0]
        energies_hea = results_hea[key_hea]['energies_per_iter'][0]

        energies_per_type_per_error[i] = [energies_uccsd, energies_hea]

# make sure to have a figs/ folder to save the plots and an out/results/ folder to save the
# results .txt files before running this script
make_convergence_plots_per_param(
        np.arange(niters_list[0]),
        energies_per_type_per_error,
        params=error_scalings,
        param_name='Depolarizing error scaling',
        fci_energy=fci_energy,
        filename=f'LiH_UCCSD_vs_HEA_noiseless_vs_noisy'
        )
