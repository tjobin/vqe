from qiskit.quantum_info import SparsePauliOp
import numpy as np
from qiskit import transpile
from qiskit.providers.fake_provider import GenericBackendV2




geometries = {
        'LiH' : 'Li 0 0 0; H 0 0 1.595',    # equilibrium geometry of LiH molecule
        'H2' : 'H 0 0 0; H 0 0 0.741;', # equilibrium geometry of H2 molecule
    }

def make_geometry(atomic_symbol):

    """
    Returns the geometry string for a given molecule, to be used in the PySCFDriver
        Args:
            - atomic_symbol: str, either 'H2' or 'LiH'
        Returns:
            - geometry: str, geometry string for the molecule
    """
    geometry = geometries[atomic_symbol]
    return geometry

def get_circuit_depth(
        state,
        hardware='ibm',
        filename='default_filename.pdf'
        ):

    """
    Returns the depth of the circuit corresponding to the given state, after transpiling it to
    the basis gates of the specified hardware and saves the transpiled circuit in a .pdf file in
    the circuits/ folder
        Args:
            - state: qiskit.circuit.QuantumCircuit, the quantum circuit whose depth we want to compute
            - hardware: str, name of the hardware to which we want to transpile the circuit; only 'ibm' supported

        Returns:
            - depth: int, depth of the transpiled circuit
    """

    if hardware == 'ibm':
        basis_gates = ['id', 'rz', 'sx', 'x', 'cx']
    else:
        raise ValueError("Hardware not supported; must be 'ibm'")

    backend = GenericBackendV2(num_qubits=state.num_qubits, basis_gates=basis_gates)

    frozen_circuit = state.decompose() 

    # Sometimes one decompose isn't enough for UCCSD (it has layers)
    # We repeat until we see standard gates
    while 'PauliEvolutionGate' in [inst.operation.name for inst in frozen_circuit.data]:
        frozen_circuit = frozen_circuit.decompose()
        # Get circuit depth on an example hardware (with gates CZ, ID, RZ, X, SX)
    transpiled_ansatz = transpile(frozen_circuit, basis_gates=basis_gates, backend=backend, optimization_level=2, seed_transpiler=0)
    transpiled_ansatz.draw(output='mpl', filename=f'circuits/{filename}.pdf')

    return transpiled_ansatz.depth()

  
