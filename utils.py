from qiskit.quantum_info import SparsePauliOp
import numpy as np
from qiskit import transpile
from qiskit.providers.fake_provider import GenericBackendV2




geometries = {
        'LiH' : 'Li 0 0 0; H 0 0 1.595',
        'dimer' : [
            ('C',(15.2590006687, 16.8121532299, 15.9738065311)),
            ('O',(9.4486299386, 13.5060171431, 14.45823873)),
            ('H',(15.4105888182, 17.5070205934, 17.9029011803)),
            ('H',(15.2544898927, 18.4025938643, 14.6708461317)),
            ('H',(16.8591544947, 15.5899710576, 15.5578683925)),
            ('H',(13.5117713588, 15.749029294,  15.7636104197)),
            ('H',(7.68753004, 13.1203202896, 14.2934357266)),
            ('H',(10.167173679, 13.0333702175, 12.8652242888))
            ],
        'dimer_' : [
            ('C', (29.448623, 17.819358, 18.344269)),
            ('H', (30.915398, 19.144469, 18.909845)),
            ('H', (27.708179, 18.253679, 19.348981)),
            ('H', (29.128138, 17.975003, 16.319352)),
            ('H', (30.042775, 15.904281, 18.798897226)),
            ('O', (9.448630, 12.577577, 12.030276)),
            ('H', (10.973795, 12.307386, 11.093119)),
            ('H', (8.430457, 11.125273, 11.667596))
            ],
        'CH4' : [
            ('C', (29.448623, 17.819358, 18.344269)),
            ('H', (30.915398, 19.144469, 18.909845)),
            ('H', (27.708179, 18.253679, 19.348981)),
            ('H', (29.128138, 17.975003, 16.319352)),
            ('H', (30.042775, 15.904281, 18.798897226))
            ],
        'H2O' : [
            ('O', (9.4486299386, 12.5775758681, 12.0302753173)),
            ('H', (10.9737937682, 12.3073847361, 11.093118626)),
            ('H', (8.4304569148,  11.1252723137, 11.6675953263))
        ],
        'Ga' : [('Ga', (0, 0, 0))],
        'Kr' : [('Kr', (0, 0, 0))],
        'Sc' : [('Sc', (0, 0, 0))],
        'Li' : [('Li', (0, 0, 0))],
        'Be' : [('Be', (0, 0, 0))],
        'B' : [('B', (0, 0, 0))],
        'C' : [('C', (0, 0, 0))],
        'H' : [('H', (0, 0, 0))],
        'H2' : 'H 0 0 0; H 0 0 0.7410102132613643;',

        'ScO' : [
            ('Sc', (0, 0, 0)),
            ('O', (0, 0, 3.2088))
            ],
        'BeH2' : [
            ('Be', (0, 0, 0)),
            ('H', (0, 0, 2.5303433)),
            ('H', (0, 0, -2.5303433))
            ],
        
    }

def make_geometry(name_mol):
    return geometries[name_mol]

def get_exact_fci_energy(hamiltonian: SparsePauliOp):
    """
    Computes the exact ground state energy (FCI) by diagonalizing the Hamiltonian.
    """
    # 1. Convert SparsePauliOp to a dense NumPy matrix
    # This works well for small systems (H2, LiH, < 12 qubits)
    H_matrix = hamiltonian.to_matrix()
    
    # 2. Use NumPy to find eigenvalues (eigh is for Hermitian matrices)
    eigenvalues = np.linalg.eigvalsh(H_matrix)
    
    # 3. The smallest eigenvalue is the Ground State Energy
    fci_energy = eigenvalues[0]
    
    return fci_energy

def get_circuit_depth(state, hardware='ibm'):

    if hardware == 'ibm':
        basis_gates = ['id', 'rz', 'sx', 'x', 'cx']
    else:
        basis_gates = ['id', 'r', 'ry', 'rx', 'rxx']

    backend = GenericBackendV2(num_qubits=state.num_qubits, basis_gates=basis_gates)

    frozen_circuit = state.decompose() 

    # Sometimes one decompose isn't enough for UCCSD (it has layers)
    # We repeat until we see standard gates
    while 'PauliEvolutionGate' in [inst.operation.name for inst in frozen_circuit.data]:
        frozen_circuit = frozen_circuit.decompose()
        # Get circuit depth on an example hardware (with gates CZ, ID, RZ, X, SX)
    transpiled_ansatz = transpile(frozen_circuit, basis_gates=basis_gates, backend=backend, optimization_level=2, seed_transpiler=0)
    transpiled_ansatz.draw(output='mpl', filename='circuits/circuit.pdf')

    return transpiled_ansatz.depth()

  
