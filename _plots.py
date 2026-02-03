import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D
from qiskit.quantum_info import entropy
from qiskit.quantum_info import partial_trace, Statevector
import numpy as np
import matplotlib.ticker as ticker


plt.rcParams.update({
    'font.size': 14,          # General font size
    'axes.labelsize': 16,     # X and Y labels
    'axes.titlesize': 18,     # Title
    'xtick.labelsize': 14,    # X-axis tick labels
    'ytick.labelsize': 14,    # Y-axis tick labels
    'legend.fontsize': 16,    # Legend
})
colors=[mcolors.TABLEAU_COLORS['tab:blue'],
        mcolors.TABLEAU_COLORS['tab:orange'],
        mcolors.TABLEAU_COLORS['tab:green'],
        mcolors.TABLEAU_COLORS['tab:red'],
        mcolors.TABLEAU_COLORS['tab:purple'],
        mcolors.TABLEAU_COLORS['tab:brown']]

def make_convergence_plots_per_param(
        iters,
        energies_per_type_per_param, # list of sublists, with each sublist containing the energies for a given param
        params, # list
        param_name,
        fci_energy,
        labels=['UCCSD', 'EfficientSU2'],
        markers=['o', '^'],
        filename='default_filename.pdf'):
    """
    Saves a figure with 2 x m convergence subplots for different values of a given parameter in a figs/ folder
        Args:
            - iters: list, contains the integers [0, 1, 2, ..., N_iters-1]
            - energies_per_type_per_param: list, of N_params sublists, of N_types subsublists, where each sublist corresponds
                to a given value of the parameter and the subsublists each correspond to a type of ansatz
            - params: list, contains the different values of the parameter of interest
            - param_name: str, is the name of the parameter of interest
            - fci_energy: float, reference energy
            - labels: list, of N_types strings, where each string is the name of an ansatz
            - markers: list, of N_types strings, where each string is the marker associated to an ansatz
            - filename: str, name of the .pdf file to be saved
        Returns:
            Nothing
    """
    ncols = (len(params) + 1) // 2  # Calculate number of columns for 2 rows
    nrows = 2  # Fixed number of rows
    fig = plt.figure(figsize=(4 * ncols, 4 * nrows))  # Adjust figure size for m x n grid
    gs = fig.add_gridspec(nrows, ncols, wspace=0.1, hspace=0.3)  # m x n grid with spacing
    axs = gs.subplots(sharex=True, sharey=True)
    colors = ['tab:blue', 'tab:orange']

    for i, param in enumerate(params):
        row, col = divmod(i, ncols)  # Determine row and column for the grid
        ax = axs[row, col]  # Access subplot based on row and column
        ax.set_title(rf'{param_name} = {param}')  # Add title for each subplot
        ax.set_xlabel('Iterations')
        if col == 0:  # Add y-axis label only for the first column
            ax.set_ylabel('Energy (Ha)')

        for j, energies in enumerate(energies_per_type_per_param[i]):
            ax.plot(
                np.concatenate([iters, [len(iters),]])[::50], # plots every 50th iteration
                energies[50::3*50], # assuming 3 circuit iterations per optimization step (SPSA)
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
    fig.legend(handles, labels, loc='lower center', ncol=len(labels), bbox_to_anchor=(0.5, -0.05))
    plt.savefig(f'figs/{filename}.pdf', bbox_inches='tight')


def make_pes_plot(
        distances,
        energies_per_type, # list of lists
        fci_energies,
        labels = ['UCCSD', 'EfficientSU2', 'Exact FCI'],
        markers=['o', '^'],
        filename='default_filename.pdf'
                  ):
    fig, ax1 = plt.subplots()

    # 2. Create the second axis (Time Steps) sharing the same X-axis
    # ax2 = ax1.twinx()
    ax1.set_xlabel('Bond distance [Å]')
    ax1.set_ylabel('Energy [Ha]', color=mcolors.TABLEAU_COLORS['tab:blue'])
    for i, energies in enumerate(energies_per_type):
        if i == 0:
            ax1.plot(distances, energies, label=labels[i], marker=markers[i], alpha=0.7, markersize=8, markeredgewidth=1.5,linestyle=None, linewidth=0, color='tab:blue')
        elif i == 1:
            ax1.plot(distances, energies, label=labels[i], marker=markers[i], alpha=0.7, markersize=8, markeredgewidth=1.5,linestyle=None, linewidth=0, color='tab:orange')
    ax1.plot(distances, fci_energies, label=labels[-1], alpha=0.7, markersize=0, markeredgewidth=0,linestyle='--', color='k')

    h1, l1 = ax1.get_legend_handles_labels()
    ax1.legend(h1, l1, loc='lower right')

    fig.tight_layout()
    plt.savefig(f'figs/{filename}.pdf')
    plt.close(fig)

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
    # ax.axhline(0.0313, color='k', label='Exact, He')
    ax.set_ylabel('Entanglement entropy')
    ax.set_xlabel('Bond distance [Å]')
    ax.legend(loc='best')
    fig.tight_layout()
    plt.savefig(f'figs/{filename}.pdf')
    plt.close(fig)


def make_pes_plots_per_nshots(energies_per_nshots_per_type, distances, nshots_list, fci_energies, filename):
    fig = plt.figure(figsize=(3 * len(nshots_list), 4))
    gs = fig.add_gridspec(1, len(nshots_list), wspace=0)
    axs = gs.subplots(sharex=True,sharey=True)
    # fig, axs = plt.subplots(1, len(nshots_list), figsize=(6 * len(nshots_list), 4), sharey=True)  # Horizontal layout with shared y-axis
    for i, nshots in enumerate(nshots_list):
        energies_per_type = [energies_per_nshots_per_type[i][j] for j in range(len(energies_per_nshots_per_type[i]))]
        ax = axs[i] if len(nshots_list) > 1 else axs  # Handle case where nshots_list has only one element
        ax.set_title(f'{nshots} shots')  # Add title for each subplot
        ax.set_xlabel('Bond distance [Å]')
        if i == 0:
            ax.set_ylabel('Energy [Ha]')

        for j, energies in enumerate(energies_per_type):
            if j == 0:
                ax.plot(distances, energies, label='UCCSD', marker='o', alpha=0.7, markersize=8, markeredgewidth=1.5, linestyle=None, linewidth=0, color='tab:blue')
            elif j == 1:
                ax.plot(distances, energies, label='EfficientSU2', marker='o', alpha=0.7, markersize=8, markeredgewidth=1.5, linestyle=None, linewidth=0, color='tab:orange')
        ax.plot(distances, fci_energies, label='Exact FCI', alpha=0.7, markersize=0, markeredgewidth=0, linestyle='--', color='k')

    # Create a common legend
    handles, labels = axs[-1].get_legend_handles_labels() if len(nshots_list) > 1 else axs.get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=len(labels), bbox_to_anchor=(0.5, -0.05))
    plt.subplots_adjust(wspace=0)  # Remove horizontal space between subplots
    plt.tight_layout()
    plt.savefig(f'figs/{filename}.pdf', bbox_inches='tight')

def make_pes_per_niters(
        distances,
        energies_per_type_per_niters,
        fci_energies,
        niters_list,
        labels = ['UCCSD', 'EfficientSU2', 'Exact FCI'],
        markers=['o', '^'],
        filename='default_filename.pdf'
                  ):
    fig = plt.figure(figsize=(3 * len(niters_list), 4))
    gs = fig.add_gridspec(1, len(niters_list), wspace=0)
    axs = gs.subplots(sharex=True,sharey=True)
    for i, niters in enumerate(niters_list):
        energies_per_type = [energies_per_type_per_niters[i][j] for j in range(len(energies_per_type_per_niters[i]))]
        ax = axs[i] if len(niters_list) > 1 else axs  # Handle case where niters_list has only one element
        ax.set_title(f'{niters} iterations')  # Add title for each subplot
        ax.set_xlabel('Bond distance [Å]')
        if i == 0:
            ax.set_ylabel('Energy [Ha]')

        for j, energies in enumerate(energies_per_type):
            if j == 0:
                ax.plot(distances, energies, label=labels[j], marker=markers[j], alpha=0.7, markersize=8, markeredgewidth=1.5,linestyle=None, linewidth=0, color='tab:blue')
            elif j == 1:
                ax.plot(distances, energies, label=labels[j], marker=markers[j], alpha=0.7, markersize=8, markeredgewidth=1.5,linestyle=None, linewidth=0, color='tab:orange')
        ax.plot(distances, fci_energies, label=labels[-1], alpha=0.7, markersize=0, markeredgewidth=0,linestyle='--', color='k')

    # Create a common legend
    handles, labels = axs[-1].get_legend_handles_labels() if len(niters_list) > 1 else axs.get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=len(labels), bbox_to_anchor=(0.5, -0.05))
    plt.subplots_adjust(wspace=0)  # Remove horizontal space between subplots
    plt.tight_layout()
    plt.savefig(f'figs/{filename}.pdf', bbox_inches='tight')