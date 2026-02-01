import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D
from qiskit.quantum_info import entropy
from qiskit.quantum_info import partial_trace, DensityMatrix, Statevector
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

def make_convergence_plot(
        iters,
        energies_per_type,
        fci_energy,
        filename = 'default_filename.pdf'
        ):
    labels = ['UCCSD', 'EfficientSU2']
    markers = ['o', '^']
    colors = ['tab:blue', 'tab:orange']
    fig, ax = plt.subplots()
    for i, energies in enumerate(energies_per_type):
        ax.plot(np.concatenate([iters, [len(iters),]])[::50], energies[50::3*50], markersize=8, label=labels[i], alpha=0.7, linestyle=None, marker=markers[i], linewidth=0, color=colors[i])
    # for i, energies in enumerate(energies_per_type):
    #     ax.axhline(y=energies[-1], color=colors[i], label=f'{labels[i]} energy : {energies[-1]:.5f} Ha')
    ax.axhline(y=fci_energy, color='k', label=f'Exact FCI energy', linestyle='--')


    ax.set_xlabel('Iterations')
    ax.set_ylabel('Energy (Ha)')
    ax.legend()
    fig.tight_layout()
    plt.savefig(f'figs/{filename}.pdf')


def make_convergence_plots_per_error(
        iters,
        energies_per_type_per_error,
        errors,
        fci_energy,
        labels=['UCCSD', 'EfficientSU2'],
        markers=['o', '^'],
        filename='default_filename.pdf'):
    ncols = (len(errors) + 1) // 2  # Calculate number of columns for 2 rows
    nrows = 2  # Fixed number of rows
    fig = plt.figure(figsize=(4 * ncols, 4 * nrows))  # Adjust figure size for m x n grid
    gs = fig.add_gridspec(nrows, ncols, wspace=0.1, hspace=0.3)  # m x n grid with spacing
    axs = gs.subplots(sharex=True, sharey=True)
    colors = ['tab:blue', 'tab:orange']

    for i, error in enumerate(errors):
        row, col = divmod(i, ncols)  # Determine row and column for the grid
        ax = axs[row, col]  # Access subplot based on row and column
        ax.set_title(rf'Error scaling $s_{i}$ = {error}')  # Add title for each subplot
        ax.set_xlabel('Iterations')
        if col == 0:  # Add y-axis label only for the first column
            ax.set_ylabel('Energy (Ha)')

        for j, energies in enumerate(energies_per_type_per_error[i]):
            ax.plot(np.concatenate([iters, [len(iters),]])[::50], energies[50::3*50], markersize=8, label=labels[j], alpha=0.7, linestyle=None, marker=markers[j], linewidth=0, color=colors[j])
        ax.axhline(y=fci_energy, color='k', label=f'Exact FCI energy', linestyle='--')

    # Create a common legend
    handles, labels = axs.flat[-1].get_legend_handles_labels() if len(errors) > 1 else axs.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=len(labels), bbox_to_anchor=(0.5, -0.05))
    # plt.tight_layout()
    plt.savefig(f'figs/{filename}.pdf', bbox_inches='tight')

def make_convergence_plots_per_shots(
        n_iters,
        energies_per_type_per_shots,
        shots_list,
        fci_energy,
        labels=['UCCSD', 'EfficientSU2'],
        markers=['o', '^'],
        filename='default_filename.pdf'):
    
    ncols = (len(shots_list) + 1) // 2  # Calculate number of columns for 2 rows
    nrows = 2  # Fixed number of rows
    fig = plt.figure(figsize=(4 * ncols, 4 * nrows))  # Adjust figure size for m x n grid
    gs = fig.add_gridspec(nrows, ncols, wspace=0.1, hspace=0.3)  # m x n grid with spacing
    axs = gs.subplots(sharex=True, sharey=True)
    colors = ['tab:blue', 'tab:orange']

    for i, shots in enumerate(shots_list):
        row, col = divmod(i, ncols)  # Determine row and column for the grid
        ax = axs[row, col]  # Access subplot based on row and column
        ax.set_title(f'{shots} shots')  # Add title for each subplot
        ax.set_xlabel('Iterations')
        if col == 0:  # Add y-axis label only for the first column
            ax.set_ylabel('Energy (Ha)')

        for j, energies in enumerate(energies_per_type_per_shots[i]):
            ax.plot(np.concatenate([n_iters, [len(n_iters),]])[::25], energies[50::3*25], markersize=8, label=labels[j], alpha=0.7, linestyle=None, marker=markers[j], linewidth=0, color=colors[j])
        # for j, energies in enumerate(energies_per_type_per_shots[i]):
        #     ax.axhline(y=energies[-2], color=colors[j], label=f'{labels[j]} energy : {energies[-1]:.5f} Ha')
        ax.axhline(y=fci_energy, color='k', label=f'Exact FCI energy', linestyle='--')

    # Create a common legend
    handles, labels = axs.flat[-1].get_legend_handles_labels() if len(shots_list) > 1 else axs.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=len(labels), bbox_to_anchor=(0.5, -0.05))
    plt.savefig(f'figs/{filename}.pdf', bbox_inches='tight')

def make_pes_plot(
        distances,
        energies_per_type,
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
    # ax2.set_ylabel('Iterations', color=mcolors.TABLEAU_COLORS['tab:orange'])
    ax1.tick_params(axis='y', labelcolor=mcolors.TABLEAU_COLORS['tab:blue'])
    # ax2.tick_params(axis='y', labelcolor=mcolors.TABLEAU_COLORS['tab:orange'])
    for i, energies in enumerate(energies_per_type):
        if i == 0:
            ax1.plot(distances, energies, label=labels[i], marker=markers[i], alpha=0.7, markersize=8, markeredgewidth=1.5,linestyle=None, linewidth=0, color='tab:blue')
            # ax2.plot(distances,  timesteps_required, label='Iterations', marker='x', alpha=0.7, markersize=8, markeredgewidth=1.5,linestyle=None, linewidth=0, color=mcolors.TABLEAU_COLORS['tab:orange'])
        elif i == 1:
            ax1.plot(distances, energies, label=labels[i], marker=markers[i], alpha=0.7, markersize=8, markeredgewidth=1.5,linestyle=None, linewidth=0, color='tab:orange')
    ax1.plot(distances, fci_energies, label=labels[-1], alpha=0.7, markersize=0, markeredgewidth=0,linestyle='--', color='k')

    h1, l1 = ax1.get_legend_handles_labels()

    # Plot on axis 2
    # h2, l2 = ax2.get_legend_handles_labels()

    # Combine them into one legend
    # ax1.legend(h1 + h2, l1 + l2, loc='lower right')
    ax1.legend(h1, l1, loc='lower right')

    fig.tight_layout()
    plt.savefig(f'figs/{filename}.pdf')
    plt.close(fig)

def make_entropy_plot(
        distances,
        qcs,
        filename
):
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

def make_energy_dt_vs_iter_plot(
        energies_per_initdt,
        dts_per_initdt,
        init_dts,
        fci_energy,
        filename
):
    fig = plt.figure()
    gs = fig.add_gridspec(1, 4, wspace=0)
    (ax1, ax3, ax5, ax7) = gs.subplots(sharex=False,sharey=True)
    ax2 = ax1.twinx()
    ax4 = ax3.twinx()
    ax6 = ax5.twinx()
    ax8 = ax7.twinx()

    prim_ax = [ax1,ax3,ax5, ax7]
    second_ax = [ax2, ax4, ax6, ax8]
    for i, ax in enumerate(prim_ax):
        ax.set_title(r'$\Delta\tau^{(0)}$' + rf' = $10^{{{-i-1}}}$')
        ax2_p = second_ax[i]
        if i == 0:
            ax2_p.set_ylim(bottom=-0.05, top=0.6)
        ax.set_xlabel('Time steps', color='k')
        ax.set_ylabel('Energy [Ha]', color='tab:blue')
        ax2_p.set_ylabel(r'$^\Delta\tau^{(i)}$' + r' [$\hbar$/Ha]', color='tab:orange')
        ax.tick_params(axis='y', labelcolor='tab:blue')
        ax2_p.tick_params(axis='y', labelcolor='tab:orange')
        iters = np.arange(len(energies_per_initdt[i][0]))
        
        ax.plot(iters, energies_per_initdt[i][0], marker='o', alpha=0.7, markersize=8, linestyle=None, linewidth=0, color=colors[0])
        ax2_p.plot(iters,  dts_per_initdt[i][0], marker='x', alpha=0.7, markersize=8, markeredgewidth=1.5,linestyle=None, linewidth=0, color=colors[1])
        ax.label_outer()
        ax2_p.label_outer()
        ax.xaxis.set_major_locator(ticker.MultipleLocator(10))
        ax.xaxis.set_minor_locator(ticker.MultipleLocator(2))
        if i == 0:
            ax.xaxis.set_major_locator(ticker.MultipleLocator(2))
            ax.xaxis.set_minor_locator(ticker.MultipleLocator(1))
        else:
            ax.xaxis.set_major_locator(ticker.MultipleLocator(4))
            ax.xaxis.set_minor_locator(ticker.MultipleLocator(2))
        # elif i==2:
        #     ax.xaxis.set_major_locator(ticker.MultipleLocator(4))
        #     ax.xaxis.set_minor_locator(ticker.MultipleLocator(1))
         
        ax.axhline(fci_energy, linestyle='--', color='k', label='Exact energy')
        # h1, l1 = ax.get_legend_handles_labels()
        # # Plot on axis 2
        # h2, l2 = ax2.get_legend_handles_labels()
        # ax.legend(h1 + h2, l1 + l2, loc='lower right')

    ax4.sharey(ax2)
    ax6.sharey(ax4)
    ax8.sharey(ax6)
    # ax2.sharey(ax6)
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