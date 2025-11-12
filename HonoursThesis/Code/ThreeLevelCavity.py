import matplotlib.pyplot as plt
import numpy as np
from IPython.display import Image
from qutip import (Qobj, tensor, basis, create, destroy, qeye, mesolve, steadystate, expect)

########
# NOTE #
########
# This implementation is based off of Coherent Control of Quantum Fluctuations Using Cavity Electromagnetically Induced Transparency (https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.111.113602)

# Initial Arguments
kappa = 1.0 # Write all parameters in terms of kappa
W_c = 40.0 * kappa
g = 50.0 * kappa
Delta = 0.01
epsilon = 1.0 * kappa
Gamma_31 = 0.1 * kappa
Gamma_32 = 0.1 * kappa
N = 10 # Truncate the Fock basis to N=10 photons

# Atomic basis kets
atomicKet1 = basis(3, 0) # |1>
atomicKet2 = basis(3, 1) # |2>
atomicKet3 = basis(3, 2) # |3>

# Set up operators in the relevant basis
s11 = tensor(atomicKet1 * atomicKet1.dag(), qeye(N))
s22 = tensor(atomicKet2 * atomicKet2.dag(), qeye(N))
s33 = tensor(atomicKet3 * atomicKet3.dag(), qeye(N))
s13 = tensor(atomicKet1 * atomicKet3.dag(), qeye(N)) # Photon absorption 1->3
s23 = tensor(atomicKet2 * atomicKet3.dag(), qeye(N)) # Photon absorption 2->3
s31 = s13.dag() # Photon emission 3->1
s32 = s23.dag() # Photon emission 3->2
a = tensor(qeye(3), destroy(N)) # Acts on photon Fock basis as lowering operator

# Decay operators
cavity_decay = np.sqrt(2*kappa) * a
atomic_decay_13 = np.sqrt(2*Gamma_31) * s13
atomic_decay_23 = np.sqrt(2*Gamma_32) * s23
c_ops = [cavity_decay, atomic_decay_13, atomic_decay_23]

def eta(n):
    return g * np.sqrt(n/2) / W_c

def steady_evo():
    state_1_vals = []
    state_2_vals = []
    state_3_vals = []
    state_4_vals = []
    state_5_vals = []
    Delta_vals = np.linspace(-100.0, 100.0, 1000)

    # Useful basis vectors
    k_s2_n0 = tensor(atomicKet2, basis(N, 0))
    k_s3_n0 = tensor(atomicKet3, basis(N, 0))
    k_s2_n1 = tensor(atomicKet2, basis(N, 1))
    k_s3_n1 = tensor(atomicKet3, basis(N, 1))

    for Delta in Delta_vals:
        # Get system for Delta value
        # Note that we assume perfect cavity resonance (Delta_1, Delta_2 = 0)
        H = -Delta * s11 + \
            Delta * a.dag() * a + \
            epsilon * (a + a.dag()) + \
            g * (a * s31 + a.dag() * s13) + \
            W_c * (s32 + s23)

        # Compute steady state solution
        rho = steadystate(H, c_ops, method='direct')

        # Find phase for n=0 and n=1 
        expect_n0 = expect(k_s2_n0 * k_s3_n0.dag(), rho)
        phi_n0 = np.angle(expect_n0)
        expect_n1 = expect(k_s2_n1 * k_s3_n1.dag(), rho)
        phi_n1 = np.angle(expect_n1)

        # Define rotated plus minus states in full Hilbert space 
        plus_n0 =  (k_s2_n0 + np.exp(-1j*phi_n0)*k_s3_n0) / np.sqrt(2)
        minus_n0 = (k_s2_n0 - np.exp(-1j*phi_n0)*k_s3_n0) / np.sqrt(2)
        plus_n1 =  (k_s2_n1 + np.exp(-1j*phi_n1)*k_s3_n1) / np.sqrt(2)
        minus_n1 = (k_s2_n1 - np.exp(-1j*phi_n1)*k_s3_n1) / np.sqrt(2)

        # psi^(0)_1
        psi_1 = tensor(atomicKet1, basis(N, 1)) - \
                eta(1) * plus_n0 + \
                minus_n0

        # psi^(0)_2
        psi_2 = tensor(atomicKet1, basis(N, 2)) - \
                eta(2) * plus_n1 + \
                minus_n1

        # Normalization
        psi_1 = psi_1.unit()
        psi_2 = psi_2.unit()

        # Operators
        psi_1_op = psi_1 * psi_1.dag()
        psi_2_op = psi_2 * psi_2.dag()

        # Get expectation values
        exp_1 = expect(psi_1_op, rho)
        exp_2 = expect(psi_2_op, rho)
        exp_3 = expect(s11, rho)
        exp_4 = expect(s22, rho)
        exp_5 = expect(s33, rho)
        state_1_vals.append(exp_1)
        state_2_vals.append(exp_2)
        state_3_vals.append(exp_3)
        state_4_vals.append(exp_4)
        state_5_vals.append(exp_5)


    fig, (ax1, ax2) = plt.subplots(2, 1) 
    fig.tight_layout()
    ax1.plot(Delta_vals, state_1_vals, label=r'$\langle \Psi^{(0)}_1|\rho|\Psi^{(0)}_1\rangle$')
    ax1.plot(Delta_vals, state_2_vals, label=r'$\langle \Psi^{(0)}_2|\rho|\Psi^{(0)}_2\rangle$')
    ax1.set_yscale("log")
    ax1.set_xlabel('Delta')
    ax1.set_ylabel('Probability of Excitation')
    ax1.legend((r'$\langle \Psi^{(0)}_1|\rho|\Psi^{(0)}_1\rangle$', r'$\langle \Psi^{(0)}_2|\rho|\Psi^{(0)}_2\rangle$'), loc='upper right')
    ax1.set_title('CEIT state excitation probabilities versus Delta')

    ax2.plot(Delta_vals, state_3_vals, label=r'$\langle 1|\rho|1\rangle$')
    ax2.plot(Delta_vals, state_4_vals, label=r'$\langle 2|\rho|2\rangle$')
    ax2.plot(Delta_vals, state_5_vals, label=r'$\langle 3|\rho|3\rangle$')
    ax2.set_yscale("log")
    ax2.set_xlabel('Delta')
    ax2.set_ylabel('Probability of Excitation')
    ax2.legend((r'$\langle 1|\rho|1\rangle$', r'$\langle 2|\rho|2\rangle$', r'$\langle 3|\rho|3\rangle$'), loc='lower right')
    ax2.set_title('Atomic basis state excitation probabilities versus Delta')


    plt.show()