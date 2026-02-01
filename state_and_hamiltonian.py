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
    Creates either a UCCSD state or an EfficientSU2 state in the active space
    
    args: 
        state_type : 'UCCSD' or 'EfficientSU2'
        mol : 'H2' or 'LiH'
        basis_set : any from the pyscf basis set bank
        active_orb : number of active orbitals in active space
        n_elec : number of electrons used in the simulation, the rest is frozen        
    """

    driver = PySCFDriver(
        atom=geometry,
        basis=basis_set,
        charge=0,
        spin=0,
    )
    transformer = ActiveSpaceTransformer(
        num_electrons=n_elec,            # Keep 2 valence electrons
        num_spatial_orbitals=active_orb      # Keep 3 orbitals (e.g. HOMO, LUMO, LUMO+1)
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
    if state_type == 'EfficientSU2':
        state = EfficientSU2(num_qubits=reduced_problem.num_spin_orbitals,
                             initial_state=hf_state)
        
    hamiltonian_op = mapper.map(reduced_problem.hamiltonian.second_q_op())

    # Add Nuclear Repulsion Energy (constant offset usually stored separately)
    # QITE minimizes the electronic part, but to match -1.137, we add this constant.
    nuclear_repulsion = reduced_problem.hamiltonian.nuclear_repulsion_energy
    core_energy = reduced_problem.hamiltonian.constants['ActiveSpaceTransformer']

    hamiltonian_full = hamiltonian_op + SparsePauliOp(["I" * hamiltonian_op.num_qubits], coeffs=[nuclear_repulsion+core_energy])

    return state, hamiltonian_full

def get_fci_energy(
            atom : str = 'LiH',
            basis_set : str = 'sto-3g',
            active_orb : int = 2,
            n_elec : int = 2,
            mapper = JordanWignerMapper(),
            ):
    """
    Creates either a UCCSD state or an EfficientSU2 state in the active space
    
    args: 
        mol : 'H2' or 'LiH'
        basis_set : any from the pyscf basis set bank
        active_orb : number of active orbitals in active space
        n_elec : number of electrons used in the simulation, the rest is frozen        
    """
    geometry = make_geometry(atom)
    driver = PySCFDriver(
        atom=geometry,
        basis=basis_set,
        charge=0,
        spin=0,
    )
    transformer = ActiveSpaceTransformer(
        num_electrons=n_elec,            # Keep 2 valence electrons
        num_spatial_orbitals=active_orb      # Keep 3 orbitals (e.g. HOMO, LUMO, LUMO+1)
    )

    problem = driver.run()
    reduced_problem = transformer.transform(problem)

    # 1. Setup the exact classical solver
    numpy_solver = NumPyMinimumEigensolver()

    # 2. Use the GroundStateEigensolver wrapper to handle the chemistry logic
    # Note: Use the same mapper you used for VQE to ensure consistency
    exact_calc = GroundStateEigensolver(mapper, numpy_solver)

    # 3. Solve the problem (using your transformed problem from your code)
    exact_result = exact_calc.solve(reduced_problem)

    # 4. Extract the total energy (Electronic + Nuclear Repulsion)
    fci_energy = exact_result.total_energies[0]

    return fci_energy