import numpy as np
import scipy.linalg as la
from qiskit_nature.second_q.mappers import JordanWignerMapper
from qiskit_nature.second_q.drivers import PySCFDriver
from qiskit.quantum_info import Pauli
from qiskit_nature.second_q.circuit.library import HartreeFock
from qiskit.circuit import QuantumCircuit
from qiskit.circuit.library import  PauliGate
from qiskit_algorithms import TimeEvolutionProblem, TrotterQRTE
from qiskit_nature.units import DistanceUnit
from qiskit import transpile
from qiskit.quantum_info import SparsePauliOp
from qiskit.primitives import StatevectorEstimator

# --- CONFIGURATION VARIABLES ---
KRYLOV_DIMENSION = 3
TAU = 0.1
TROTT_STEPS = 5

def ev(result, pub_index: int = 0) -> float:
    """Extract a single expectation value from either V1 (.values) or V2 (pub_result.data.evs) results."""
    # Old style (Aer Estimator / V1)
    if hasattr(result, "values"):
        return float(np.asarray(result.values[pub_index]).reshape(-1)[0])

    # New style (V2 primitives)
    try:
        pub_res = result[pub_index]  # PrimitiveResult is indexable in V2
    except Exception:
        pub_res = result.pub_results[pub_index]  # fallback for some versions

    # In V2, expectation values live here
    return float(np.asarray(pub_res.data.evs).reshape(-1)[0])

def get_krylov_state_circuit(initial_state_circuit, H_op, k, tau, trotter_steps):
    """
    Constructs and returns the circuit for the k-th Krylov state |psi_k> = U(k*tau) |psi_0>.
    """

    evolution_time = k * tau

    if k == 0: # if the timestep is zero return the initial state
        unitary_circuit = initial_state_circuit.copy()
        unitary_circuit = unitary_circuit.decompose()
    else:
        evolution_problem = TimeEvolutionProblem(
            hamiltonian=H_op,
            time=evolution_time,
            initial_state=initial_state_circuit
        )
        trotter_strategy = TrotterQRTE(num_timesteps=trotter_steps)
        full_circuit = trotter_strategy.evolve(evolution_problem)
        unitary_circuit = full_circuit.evolved_state.decompose()

    return unitary_circuit


def haddy_test(num_qubits, A_i, A_j, pauli_string):
    """
    Builds the Hadamard-test circuit for Re/Im of <psi_i|P|psi_j>
    by implementing controlled-(A_i^dagger P A_j).
    """
    N = num_qubits
    qc = QuantumCircuit(N + 1)
    anc = 0
    targets = list(range(1, N + 1))

    V = QuantumCircuit(N)
    V.compose(A_i.inverse(), inplace=True)

    if pauli_string != Pauli('I' * N):
        V.append(PauliGate(pauli_string.to_label()), range(N))

    V.compose(A_j, inplace=True)

    qc.h(anc)
    qc.append(V.to_gate().control(1), [anc] + targets)
    qc.h(anc)

    return qc



def quantum_krylov_diagonalization():
    """
    Performs Quantum Krylov Diagonalization (QKD) for a molecule.
    """
    M = KRYLOV_DIMENSION
    tau = TAU
    trotter_steps = TROTT_STEPS

    # --- 1. Define the Molecule and Hamiltonian ---
    print(f"--- 1. Setting up Molecule for QKD (M={M}, tau={tau}, steps={trotter_steps}) ---")
    bond_length = 0.742
    atom_info = f"H 0 0 0; H 0 0 {bond_length}"
    driver = PySCFDriver(atom=atom_info,
                         unit=DistanceUnit.ANGSTROM,
                         basis="sto3g")
    problem = driver.run()

    # --- 2. Map to Qubit Operator and Decompose ---
    mapper = JordanWignerMapper()
    second_quantized_op = problem.hamiltonian.second_q_op()

    H_pauli_op = mapper.map(second_quantized_op)


    nuclear_repulsion_energy = problem.nuclear_repulsion_energy

    pauli_coeffs = H_pauli_op.coeffs
    pauli_terms = H_pauli_op.paulis

    num_particles = (problem.num_alpha, problem.num_beta)
    num_spatial_orbitals = problem.num_spatial_orbitals
    num_qubits = H_pauli_op.num_qubits  # N

    # --- 3. Prepare the Initial State (Hartree-Fock) ---
    print("\n--- 3. Preparing Hartree-Fock State ---")

    initial_state_circuit = HartreeFock(
        num_spatial_orbitals=num_spatial_orbitals,
        num_particles=num_particles,
        qubit_mapper=mapper
    )

    # --- 4. Pre-calculate Krylov Circuits ---
    print(f"\n--- 4. Pre-calculating {M} Krylov Circuits (U(k*tau)|HF>) ---")

    krylov_circuits = []

    for k in range(M):
        circuit_k = get_krylov_state_circuit(initial_state_circuit, H_pauli_op, k, tau, trotter_steps)
        krylov_circuits.append(circuit_k)

    # --- 5. Calculate Overlap Matrix and Projected Hamiltonian ---

    H_K = np.zeros((M, M), dtype=complex)
    S_K = np.zeros((M, M), dtype=complex)

    estimator = StatevectorEstimator()
    # estimator = Estimator()

    n_tot = num_qubits + 1  # ancilla + system

    observe_ancilla_Z = SparsePauliOp.from_sparse_list([("Z", [0], 1.0)], num_qubits=n_tot)
    observe_ancilla_Y = SparsePauliOp.from_sparse_list([("Y", [0], 1.0)], num_qubits=n_tot)

    print(f"\n--- 5. Building Scalable Krylov Subspace (Dimension M={M}) ---")

    for i in range(M):
        for j in range(i + 1):
            print(f"Measuring element ({i},{j})...")

            circuit_i = krylov_circuits[i]
            circuit_j = krylov_circuits[j]

            # --- OVERLAP MATRIX S_K[i, j] = <psi_i | I | psi_j> ---

            observable_I = Pauli('I' * num_qubits)

            qc_S = haddy_test(num_qubits, circuit_i, circuit_j, observable_I)
            qc_S_t = transpile(qc_S, optimization_level=0)
            res = estimator.run([
                (qc_S_t, observe_ancilla_Z),
                (qc_S_t, observe_ancilla_Y),
            ]).result()

            # res = estimator.run([qc_S, qc_S], [observe_ancilla_Z, observe_ancilla_Y]).result()

            S_ij_real = ev(res, 0)
            S_ij_imag = ev(res, 1)
            S_ij = S_ij_real + 1j * S_ij_imag
            print(f"S element {i},{j}: {S_ij}")

            S_K[i, j] = S_ij
            if i != j:
                S_K[j, i] = np.conj(S_ij)  # Hermiticity

            # --- HAMILTONIAN MATRIX H_K[i, j] = sum_p h_p * <psi_i| P_p |psi_j> ---

            H_ij_sum = 0.0 + 0j

            for h_p, P_p in zip(pauli_coeffs, pauli_terms):
                qc_H = haddy_test(num_qubits, circuit_i, circuit_j, P_p)
                qc_H_t = transpile(qc_H, optimization_level=0)

                resH = estimator.run([
                    (qc_H_t, observe_ancilla_Z),
                    (qc_H_t, observe_ancilla_Y),
                ]).result()
                P_p_overlap_real = ev(resH, 0)
                P_p_overlap_imag = ev(resH, 1)
                P_p_overlap = P_p_overlap_real + 1j * P_p_overlap_imag
                H_ij_sum += h_p * P_p_overlap
                print(f"Pauli sum: {P_p_overlap} Pauli: {P_p} weight: {h_p}")
            print(f"Final sum: {H_ij_sum}")
            H_K[i, j] = H_ij_sum
            if i != j:
                H_K[j, i] = np.conj(H_ij_sum)  # Hermiticity

    # --- 6. Diagonalize the Subspace Problem (Classical Final Step) ---
    print("\n--- 6. Solving Generalized Eigenvalue Problem ---")
    print("Effective Krylov Hamiltonian Matrix (H_K, Real Part):")
    print(np.round(H_K.real, 6))
    print("Effective Krylov Hamiltonian Matrix (H_K, Imaginary Part):")
    print(np.round(H_K.imag, 6))
    print("\nEffective Krylov Overlap Matrix (S_K, Real Part):")
    print(np.round(S_K.real, 6))
    print("\nEffective Krylov Overlap Matrix (S_K, Imaginary Part):")
    print(np.round(S_K.imag, 6))

    # The Generalized Eigenvalue Problem is H_K * c = E * S_K * c
    eigs = la.eig(H_K, S_K, right=False)

    # pick the minimum real part (and you may optionally filter small imaginary noise)
    ground_electronic_energy = np.min(eigs.real)
    print(f"\nCalculated Krylov Subspace Eigenvalues: {eigs}")
    print(f"QKD Electronic Ground Energy (min real part): {ground_electronic_energy:.6f} Hartree")

    # Final result (Total Energy)
    total_ground_energy = ground_electronic_energy + nuclear_repulsion_energy

    print("\n========================================================")
    print(f"TOTAL GROUND ENERGY (H2, M={M}): {total_ground_energy:.6f} Hartree")
    print("========================================================")


if __name__ == "__main__":
    quantum_krylov_diagonalization()
