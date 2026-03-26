import matplotlib.pyplot as plt
import numpy as np
from IPython.display import Image
from qutip import (Qobj, tensor, basis, create, destroy, qeye, mesolve, steadystate, expect)

# Initial Arguments
w_a = 2.0
w_c = 2.0
W = 0.02
gamma = 0.005
xi = 0.0025
eta = 0.05
h = 1.0
N = 10 # Truncate the Fock basis to N photons
times = np.linspace(0, 2500, 10000)

# Set up operators in the relevant basis
a = tensor(qeye(2), destroy(N)) # Acts on photon Fock basis with N states as lowering operator
s_minus = tensor(destroy(2), qeye(N)) # Acts on atomic basis with 2 states as lowering operator
s_atom = tensor(Qobj([[0, 0], [0, 1]]), qeye(N)) # Acts on atomic basis with a shifted ground states

# Set up the Hamiltonian
H = (h * w_a) * s_atom + (h * w_c) * a.dag() * a + (h * W)/2 * (s_minus.dag() * a + s_minus * a.dag())

# Set up dissipative operator for a lossy cavity
cavity_decay = np.sqrt(gamma) * a
c_ops = [cavity_decay]

# Set up the expectation value operators
number_operator = a.dag() * a
atomic_state_operator = s_minus.dag() * s_minus
e_ops = [number_operator, atomic_state_operator]

# Start the system in an initially excited state with 0 photons
psi_0 = tensor(basis(2,1), basis(N, 0))

# Time evolve the system using qutip mesolve
result = mesolve(H, psi_0, times, c_ops=c_ops, e_ops=e_ops)
photons = result.expect[0]
atomic = result.expect[1]

plt.plot(times, photons)
plt.plot(times, atomic)
plt.xlabel('Time')
plt.ylabel('Probability of Excitation')
plt.legend(('Cavity ', 'Atom'))
plt.title('Rabi Oscillation in Jaynes-Cummings Model')
plt.show()