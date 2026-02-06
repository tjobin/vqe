from utils import make_geometry
from qiskit.circuit.library import EfficientSU2
from qiskit_nature.second_q.drivers import PySCFDriver
from qiskit_nature.second_q.mappers import JordanWignerMapper
from qiskit_nature.second_q.circuit.library.ansatzes.uccsd import UCCSD
from qiskit_nature.second_q.circuit.library import HartreeFock
from qiskit_nature.second_q.transformers import ActiveSpaceTransformer
from qiskit.quantum_info import SparsePauliOp
from qiskit_algorithms import NumPyMinimumEigensolver
from qiskit_nature.second_q.algorithms import GroundStateEigensolver


def get_state_and_hamiltonian(
            state_type: str = 'UCCSD',
            geometry : str = 'H 0 0 0; H 0 0 0.7410102132613643;',
            basis_set : str = 'sto-3g',
            active_orb : int = 2,
            n_elec : int = 2,
            mapper = JordanWignerMapper(),
            ):
    
    """
    Returns the quantum circuit preparing the state (UCCSD or EfficientSU2) and the Hamiltonian of the system in the active space.
        Args: 
            - state_type : str, 'UCCSD' or 'EfficientSU2'
            - mol : 'H2' or 'LiH'
            - basis_set : str, any from the pyscf basis set databank, e.g. 'sto-3g'
            - active_orb : int, number of active orbitals in active space
            - n_elec : int, number of electrons used in the simulation, the rest is frozen     
            - mapper : qiskit_nature.second_q.mappers, fermion-to-qubit mapper, e.g. JordanWignerMapper() or ParityMapper()
        Returns:
            - state : qiskit.circuit.QuantumCircuit, the quantum circuit preparing the state (UCCSD or EfficientSU2)
            - hamiltonian_full : qiskit.quantum_info.SparsePauliOp, the Hamiltonian of the system in the active space,
            including the nuclear repulsion energy and core electrons energies as a constant offset
    """

    driver = PySCFDriver(
        atom=geometry,
        basis=basis_set,
        charge=0,   # neutral molecule assumed
        spin=0,     # singlet state assumed
    )
    transformer = ActiveSpaceTransformer(
        num_electrons=n_elec,            # Keep n_elec valence electrons
        num_spatial_orbitals=active_orb      # Keep active_orb orbitals (e.g. 3 --> HOMO, LUMO, LUMO+1)
    )

    problem = driver.run()
    reduced_problem = transformer.transform(problem)
    hf_state = HartreeFock(
            num_spatial_orbitals=reduced_problem.num_spatial_orbitals,
            num_particles=reduced_problem.num_particles,
            qubit_mapper=mapper
        )
    if state_type == 'UCCSD':
        state = UCCSD(
            num_spatial_orbitals=reduced_problem.num_spatial_orbitals,
            num_particles=reduced_problem.num_particles,
            qubit_mapper=mapper,
            initial_state=hf_state
            )
    elif state_type == 'EfficientSU2':
        state = EfficientSU2(num_qubits=reduced_problem.num_spin_orbitals,
                             initial_state=hf_state)
    else:
        raise ValueError("The ansatz type is unsupported. Must be 'UCCSD' or 'EfficientSU2'.")
        
        
    hamiltonian_op = mapper.map(reduced_problem.hamiltonian.second_q_op())

    # Add nuclear repulsion energy and core electrons energies
    # QITE minimizes the electronic part, but to match the values found in the literature, we add these constants.
    nuclear_repulsion = reduced_problem.hamiltonian.nuclear_repulsion_energy
    core_energy = reduced_problem.hamiltonian.constants['ActiveSpaceTransformer']

    hamiltonian_full = hamiltonian_op + SparsePauliOp(["I" * hamiltonian_op.num_qubits], coeffs=[nuclear_repulsion+core_energy])

    return state, hamiltonian_full

def get_fci_energy(
            atomic_symbol : str = 'LiH',
            basis_set : str = 'sto-3g',
            active_orb : int = 2,
            n_elec : int = 2,
            mapper = JordanWignerMapper(),
            ):
    
    """
    Returns the exact ground state energy (FCI) of the system in the active space, to be used as a reference for the VQE results.
        Args: 
            atomic_symbol : str, 'H2' or 'LiH',
            basis_set : str, any from the pyscf basis set databank, e.g. 'sto-3g'
            active_orb : int, number of active orbitals in active space
            n_elec : int, number of electrons used in the simulation, the rest is frozen    
            mapper : qiskit_nature.second_q.mappers, fermion-to-qubit mapper, e.g. JordanWignerMapper() or ParityMapper()
        Returns:
            fci_energy : float, the exact ground state energy (FCI) of the system in the active space
    """

    geometry = make_geometry(atomic_symbol)
    driver = PySCFDriver(
        atom=geometry,
        basis=basis_set,
        charge=0,   # neutral molecule assumed
        spin=0,     # singlet state assumed
    )
    transformer = ActiveSpaceTransformer(
        num_electrons=n_elec,            # Keep n_elec valence electrons
        num_spatial_orbitals=active_orb      # Keep active_orb orbitals (e.g. 3 --> HOMO, LUMO, LUMO+1)
    )

    problem = driver.run()
    reduced_problem = transformer.transform(problem)

    # Setup the exact classical solver
    numpy_solver = NumPyMinimumEigensolver()

    # Use the GroundStateEigensolver wrapper to handle the chemistry logic
    # Note: Use the same mapper you used for VQE to ensure consistency
    exact_calc = GroundStateEigensolver(mapper, numpy_solver)

    # Solve the problem (using your transformed problem from your code)
    exact_result = exact_calc.solve(reduced_problem)

    # Extract the total energy (Electronic + Nuclear Repulsion)
    fci_energy = exact_result.total_energies[0]

    return fci_energy