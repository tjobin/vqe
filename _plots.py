import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from qiskit.quantum_info import entropy
from qiskit.quantum_info import partial_trace, Statevector
import numpy as np


plt.rcParams.update({
    'font.size': 12,          # General font size
    'axes.labelsize': 12,     # X and Y labels
    'axes.titlesize': 14,     # Title
    'xtick.labelsize': 12,    # X-axis tick labels
    'ytick.labelsize': 12,    # Y-axis tick labels
    'legend.fontsize': 12,    # Legend
})
colors=[mcolors.TABLEAU_COLORS['tab:blue'],
        mcolors.TABLEAU_COLORS['tab:orange'],
        mcolors.TABLEAU_COLORS['tab:green'],
        mcolors.TABLEAU_COLORS['tab:red'],
        mcolors.TABLEAU_COLORS['tab:purple'],
        mcolors.TABLEAU_COLORS['tab:brown']]

def make_convergence_plots_per_param(
        iters,
        energies_per_type_per_param, 
        params,
        param_name,
        fci_energy,
        labels = ['UCCSD', 'EfficientSU2'],
        markers = ['o', '^'],
        filename = 'default_filename.pdf'):
    
    """
    Saves a figure with 1 x 2 or 2 x N_params / 2 convergence subplots for different values of a given parameter in a figs/ folder
        Args:
            - iters: list, contains the integers [0, 1, 2, ..., N_iters-1]
            - energies_per_type_per_param: list, of N_params sublists, of N_types subsublists, of len N_iters, contains the energies
            at each value of the parameter, for each type of ansatz, and at each iteration
            - params: list, contains the different values of the parameter of interest
            - param_name: str, is the name of the parameter of interest
            - fci_energy: float, reference energy
            - labels: list, of N_types strings, where each string is the name of an ansatz
            - markers: list, of N_types strings, where each string is the marker associated to an ansatz
            - filename: str, name of the .pdf file to be saved
        Returns:
            Nothing
    """
    if len(params) <= 2:
        nrows = 1
        ncols = len(params)
    else:
        nrows = 2  # Fixed number of rows
        ncols = (len(params) + 1) // 2  # Calculate number of columns for 2 rows
    fig = plt.figure(figsize=(4 * ncols, 4 * nrows))  # Adjust figure size for m x n grid
    gs = fig.add_gridspec(nrows, ncols, wspace=0.1, hspace=0.3)  # m x n grid with spacing
    axs = gs.subplots(sharex=True, sharey=True)

    for i, param in enumerate(params):
        if len(params) <= 2:
            row = 0
            col = i
            ax = axs[i]
        else:
            row, col = divmod(i, ncols)  # Determine row and column for the grid
            ax = axs[row, col] if col > 1 else axs[row]  # Access subplot based on row and column
        ax.set_title(rf'{param_name} = {param}')  # Add title for each subplot
        ax.set_xlabel('Iterations')
        if col == 0:  # Add y-axis label only for the first column
            ax.set_ylabel('Energy (Ha)')

        for j, energies in enumerate(energies_per_type_per_param[i]):
            ax.plot(
                np.concatenate([iters, [len(iters),]])[::5], # plots every 5th iteration
                energies[50::3*5], # assuming 3 circuit evaluations per optimization step (SPSA with blocking)
                markersize=8,
                label=labels[j],
                alpha=0.7,
                linestyle=None,
                marker=markers[j],
                linewidth=0,
                color=colors[j]
                )
        ax.axhline(y=fci_energy, color='k', label=f'Exact FCI energy', linestyle='--')

    # Create a common legend
    handles, labels = axs.flat[-1].get_legend_handles_labels() if len(params) > 1 else axs.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=len(labels), bbox_to_anchor=(0.5, -0.15))
    plt.savefig(f'figs/{filename}.pdf', bbox_inches='tight')

def make_entropy_plot(
        distances,
        qcs,
        filename
    ):

    """
    Saves a figure with a single subplot of the entanglement entropy of qubit 0 and 2 at N_dist
    different bond distances in a figs/ folder
        Args:
            - distances: list, of len N_dist, contains the bond lengths
            - qcs: list, of len N_dist, contains the converged states at the different bond lengths
            - filename: str, name of the .pdf file to be saved
        Returns:
            Nothing
    """  

    states = [Statevector(qc) for qc in qcs]
    rhos_q0 = [partial_trace(state, [1,3]) for state in states]
    entropies = [entropy(rho) for rho in rhos_q0]
    fig, ax = plt.subplots()
    ax.plot(distances, entropies, marker='o', label=r'Adaptive QITE', alpha=0.7, markersize=8, markeredgewidth=1.5,linestyle='-.')
    ax.set_ylabel('Entanglement entropy')
    ax.set_xlabel('Bond distance [Å]')
    ax.legend(loc='best')
    plt.savefig(f'figs/{filename}.pdf', bbox_inches='tight')
    plt.close(fig)


def make_pes_plots_per_param(
        distances,
        energies_per_type_per_param, 
        params,
        param_name,
        fci_energies,
        labels=['UCCSD', 'EfficientSU2'],
        markers=['o', '^'],
        filename='default_filename.pdf'
        ):
    
    """
    Saves a figure with 1 x N_params potential energy surface subplots for N_types different ansatze
    for different values of a given parameter in a figs/ folder
        Args:
            - distances: list, of N_dist float, contains each bond length at which the PES is evaluated
            - energies_per_type_per_param: list, of N_params sublists, of N_types subsublists, contains the energies at 
            each value of the parameter for each type of ansatz at each bond length
            - params: list, of N_params numbers, contains the different values of the parameter of interest
            - param_name: str, is the name of the parameter of interest
            - fci_energies: list, of N_dist numbers, contains the reference energy at each bond length
            - labels: list, of N_types strings, where each string is the name of an ansatz
            - markers: list, of N_types strings, where each string is the marker associated to an ansatz
            - filename: str, name of the .pdf file to be saved
        Returns:
            Nothing
    """

    fig = plt.figure(figsize=(3 * len(params), 4))
    gs = fig.add_gridspec(1, len(params), wspace=0)
    axs = gs.subplots(sharex=True,sharey=True)
    # fig, axs = plt.subplots(1, len(nshots_list), figsize=(6 * len(nshots_list), 4), sharey=True)  # Horizontal layout with shared y-axis
    for i, param in enumerate(params):
        energies_per_type = [energies_per_type_per_param[i][j] for j in range(len(energies_per_type_per_param[i]))]
        ax = axs[i] if len(params) > 1 else axs  # Handle case where nshots_list has only one element
        ax.set_title(f'{param_name} =  {param}')  # Add title for each subplot
        ax.set_xlabel('Bond distance [Å]')
        if i == 0:
            ax.set_ylabel('Energy [Ha]')

        for j, energies in enumerate(energies_per_type):
            ax.plot(distances, energies, label=labels[j], marker=markers[j], alpha=0.7, markersize=8, markeredgewidth=1.5, linestyle=None, linewidth=0, color=colors[j])
        ax.plot(distances, fci_energies, label='Exact FCI', alpha=0.7, markersize=0, markeredgewidth=0, linestyle='--', color='k')

    # Create a common legend
    handles, labels = axs[-1].get_legend_handles_labels() if len(params) > 1 else axs.get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=len(labels), bbox_to_anchor=(0.5, -0.15))
    plt.subplots_adjust(wspace=0)  # No horizontal space between subplots
    plt.tight_layout()
    plt.savefig(f'figs/{filename}.pdf', bbox_inches='tight')