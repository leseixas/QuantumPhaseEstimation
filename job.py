#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
  H2 molecule with Quantum Phase Estimation (QPE) algorithm
'''

from collections import OrderedDict
import time

from qiskit import Aer
from qiskit.transpiler import PassManager
from qiskit.aqua import AquaError
from qiskit.aqua import QuantumInstance
from qiskit.aqua.algorithms import ExactEigensolver
from qiskit.aqua.algorithms import QPE
from qiskit.aqua.components.iqfts import Standard
from qiskit.chemistry import FermionicOperator
from qiskit.chemistry import QiskitChemistry
from qiskit.chemistry.aqua_extensions.components.initial_states import HartreeFock
from qiskit.chemistry.drivers import PySCFDriver, UnitsType

t_initial = time.time()

distance = 0.735
driver = PySCFDriver(atom='H .0 .0 .0; H .0 .0 {}'.format(distance),
                     unit=UnitsType.ANGSTROM, charge=0, spin=0, basis='sto3g')
molecule = driver.run()

qubit_mapping = 'jordan_wigner'
fer_op = FermionicOperator(h1=molecule.one_body_integrals, h2=molecule.two_body_integrals)
qubit_op = fer_op.mapping(map_type=qubit_mapping,threshold=1e-10).two_qubit_reduced_operator(2)

exact_eigensolver = ExactEigensolver(qubit_op, k=1)
result_ee = exact_eigensolver.run()
reference_energy = result_ee['energy']
print('The exact ground state energy is: {} eV'.format(result_ee['energy']*27.21138506))

num_particles = molecule.num_alpha + molecule.num_beta
two_qubit_reduction = (qubit_mapping == 'parity') 
num_orbitals = qubit_op.num_qubits + (2 if two_qubit_reduction else 0)

print('Number of qubits: ',qubit_op.num_qubits)

num_time_slices = 50
n_ancillae = 8
print('Number of ancillae:', n_ancillae)

state_in = HartreeFock(qubit_op.num_qubits, num_orbitals,
                       num_particles, qubit_mapping, two_qubit_reduction)
iqft = Standard(n_ancillae)

qpe = QPE(qubit_op, state_in, iqft, num_time_slices, n_ancillae,
          expansion_mode='suzuki',
          expansion_order=2, shallow_circuit_concat=True)

# backend
#backend = Aer.get_backend('qasm_simulator')

# IBM Q
from qiskit import IBMQ
provider0 = IBMQ.load_account()
#large_enough_devices = IBMQ.backends(filters=lambda x: x.configuration().n_qubits > qubit_op.num_qubits and not x.configuration().simulator)
backend = provider0.get_backend('ibmq_16_melbourne')
#ibmq_16_melbourne
#ibmqx2
#ibmqx4
#ibmq_qasm_simulator

print("Backend:", backend)

quantum_instance = QuantumInstance(backend, shots=100, pass_manager=PassManager())
result_qpe = qpe.run(quantum_instance)
t_final = time.time()
print('The ground state energy as computed by QPE is: {} eV'.format(result_qpe['energy']*27.21138506))
print("...")
print('Difference: {}'.format((result_qpe['energy']-result_ee['energy'])*27.21138506))
print(result_qpe)
print("Run time: {:.2f} seconds".format(t_final-t_initial))
