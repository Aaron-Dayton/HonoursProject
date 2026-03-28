import matplotlib.pyplot as plt
import numpy as np
import random
from IPython.display import Image
from qutip import (Qobj, tensor, basis, create, destroy, qeye, mesolve, steadystate, expect, num)

##########################################
# CONSTANTS, BASIS STATES, AND OPERATORS #
##########################################

# Constants
hbar = 1
epsilon_0 = 1
c = 1
n = 1 # Index of refraction for Rubidium
boltzmann_const = 1
Rb_mass = 1
kappa = 1.0 # Write all parameters relative to kappa (cavity decay rate)
Gamma_12 = 0.1 * kappa
Gamma_13 = 0.1 * kappa
Gamma_24 = 0.05 * kappa
Gamma_34 = 0.05 * kappa
N_Fock = 5 # Truncate each Fock basis to N_Fock photons
d_12 = 1
d_34 = 1
omega_12 = 1 / 795
omega_13 = 1 / 780
omega_24 = 1 / 1475
omega_34 = 1 / 1530

# Atomic Basis Kets
atomicKet1 = basis(4, 0) # |1>
atomicKet2 = basis(4, 1) # |2>
atomicKet3 = basis(4, 2) # |3>
atomicKet4 = basis(4, 3) # |4>

atomic_density = lambda temp: 1 / temp * 10**(31.178 - 4040 / temp)


# Set up operators in the relevant basis
S11 = tensor(atomicKet1 * atomicKet1.dag(), qeye(N_Fock), qeye(N_Fock))
S22 = tensor(atomicKet2 * atomicKet2.dag(), qeye(N_Fock), qeye(N_Fock))
S33 = tensor(atomicKet3 * atomicKet3.dag(), qeye(N_Fock), qeye(N_Fock))
S44 = tensor(atomicKet4 * atomicKet4.dag(), qeye(N_Fock), qeye(N_Fock))

S12 = tensor(atomicKet1 * atomicKet2.dag(), qeye(N_Fock), qeye(N_Fock)) # Photon emission 1->2
S13 = tensor(atomicKet1 * atomicKet3.dag(), qeye(N_Fock), qeye(N_Fock)) # Photon emission 1->3
S24 = tensor(atomicKet2 * atomicKet4.dag(), qeye(N_Fock), qeye(N_Fock)) # Photon emission 2->4
S34 = tensor(atomicKet3 * atomicKet4.dag(), qeye(N_Fock), qeye(N_Fock)) # Photon emission 3->4

S21 = S12.dag() # Photon absorption 2->1
S31 = S13.dag() # Photon absorption 3->1
S42 = S24.dag() # Photon absorption 4->2
S43 = S34.dag() # Photon absorption 4->3

# a and b act on photon independent Fock bases as lowering operator
a = tensor(qeye(4), destroy(N_Fock), qeye(N_Fock))
b = tensor(qeye(4), qeye(N_Fock),    destroy(N_Fock))

# Creation operators
a_dag = a.dag()
b_dag = b.dag()

# Number Operator for field a
num_a_op = tensor(qeye(4), num(N_Fock),  qeye(N_Fock))
num_b_op = tensor(qeye(4), qeye(N_Fock), num(N_Fock))

# Decay operators
cavity_a_decay = np.sqrt(2*kappa) * a
cavity_b_decay = np.sqrt(2*kappa) * a
atomic_decay_12 = np.sqrt(2*Gamma_12) * S12
atomic_decay_13 = np.sqrt(2*Gamma_13) * S13
atomic_decay_24 = np.sqrt(2*Gamma_24) * S24
atomic_decay_34 = np.sqrt(2*Gamma_34) * S34
c_ops = [
    cavity_a_decay,
    cavity_b_decay,
    atomic_decay_12,
    atomic_decay_13,
    atomic_decay_24,
    atomic_decay_34
]




################
# HAMILTONIANS #
################

def form_Hamiltonian_ab(
        Omega_s,  # Signal Rabi frequency (Related to electric field strength)
        Omega_i,  # Idler Rabi frequency (Related to electric field strength)
        Omega_I,  # Pump 1 Rabi Frequency (Related to electric field strength)
        Omega_II, # Pump 2 Rabi Frequency (Related to electric field strength)
        Delta_s,  # Signal detuning
        Delta_I,  # Pump 1 detuning
        Delta_II, # Pump 2 detuning
        Delta_a,  # Cavity a detuning
        Delta_b   # Cavity b detuning
    ):

    H_atom = hbar * (
        Delta_I * S22 +
        Delta_s * S33 + 
        (Delta_II + Delta_s) * S44
    )

    H_photon = hbar * (
        Delta_a * a_dag * a +
        Delta_b * b_dag * b
    )

    H_int = hbar * (
        Omega_s    * a_dag * S12 + np.conj(Omega_s)  * a * S21 +
        Omega_I    * S13         + np.conj(Omega_I)  * S31     +
        Omega_II   * S24         + np.conj(Omega_II) * S42     +
        Omega_i    * b_dag * S34 + np.conj(Omega_i)  * b * S43
    )

    epsilon_s = hbar * Omega_s / d_12
    epsilon_i = hbar * Omega_i / d_34

    H_noise = hbar * (
        epsilon_s * a + np.conj(epsilon_s) * a_dag +
        epsilon_i * b + np.conj(epsilon_i) * b_dag
    )

    H_total = H_atom + H_photon + H_int + H_noise
    return H_total




def form_Hamiltonian_a(
        Omega_s,  # Signal coupling (Related to electric field strength)
        Omega_i,  # Idler coupling (Related to electric field strength)
        Omega_I,  # Pump 1 Rabi Frequency (Related to electric field strength)
        Omega_II, # Pump 2 Rabi Frequency (Related to electric field strength)
        Delta_s,  # Signal detuning
        Delta_I,  # Pump 1 detuning
        Delta_II, # Pump 2 detuning
        Delta_a,  # Cavity a detuning
        Delta_b   # Cavity b detuning
    ):

    H_atom = hbar * (
        Delta_I * S22 +
        Delta_s * S33 + 
        (Delta_II + Delta_s) * S44
    )

    H_photon = hbar * (
        Delta_a * a_dag * a +
        Delta_b * b_dag * b
    )

    H_int = hbar/2 * (
        Omega_s  * a_dag * S12 + np.conj(Omega_s)  * a * S21 +
        Omega_I  * S13         + np.conj(Omega_I)  * S31     +
        Omega_II * S24         + np.conj(Omega_II) * S42     +
        Omega_i  * S34         + np.conj(Omega_i)  * S43
    )

    epsilon_s = hbar * Omega_s / d_12

    H_noise = hbar * (
        epsilon_s * a + np.conj(epsilon_s) * a_dag 
    )

    H_total = H_atom + H_photon + H_int + H_noise
    return H_total



def form_Hamiltonian_b(
        Omega_s,  # Signal coupling (Related to electric field strength)
        Omega_i,  # Idler coupling (Related to electric field strength)
        Omega_I,  # Pump 1 Rabi Frequency (Related to electric field strength)
        Omega_II, # Pump 2 Rabi Frequency (Related to electric field strength)
        Delta_s,  # Signal detuning
        Delta_I,  # Pump 1 detuning
        Delta_II, # Pump 2 detuning
        Delta_a,  # Cavity a detuning
        Delta_b   # Cavity b detuning
    ):

    H_atom = hbar * (
        Delta_I * S22 +
        Delta_s * S33 + 
        (Delta_II + Delta_s) * S44
    )

    H_photon = hbar * (
        Delta_a * a_dag * a +
        Delta_b * b_dag * b
    )

    H_int = hbar/2 * (
        Omega_s  * S12         + np.conj(Omega_s)  * S21     +
        Omega_I  * S13         + np.conj(Omega_I)  * S31     +
        Omega_II * S24         + np.conj(Omega_II) * S42     +
        Omega_i  * b_dag * S34 + np.conj(Omega_i)  * b * S43
    )

    epsilon_i = hbar * Omega_i / d_34

    H_noise = hbar * (
        epsilon_i * b + np.conj(epsilon_i) * b_dag
    )

    H_total = H_atom + H_photon + H_int + H_noise
    return H_total



####################
# SOLUTION METHODS #
####################

def Maxwell_Bloch_step(rho, Omega_s, Omega_i, Delta_s, Delta_i, temp, step_size):
    rho_12 = rho[1,2]
    rho_34 = rho[3,4]
    
    # NOTE - We are using c here instead of group velocity here
    #        So this is an approximation
    d_z_Omega_s = 1j * (omega_12 + Delta_s) * d_12**2 * atomic_density(temp) * rho_12 / (2 * c * epsilon_0 * hbar)
    d_z_Omega_i = 1j * (omega_34 + Delta_i) * d_34**2 * atomic_density(temp) * rho_34 / (2 * c * epsilon_0 * hbar)

    Omega_s_step = d_z_Omega_s * step_size
    Omega_i_step = d_z_Omega_i * step_size

    # Update rabi frequencies
    new_Omega_s = Omega_s + Omega_s_step
    new_Omega_i = Omega_i + Omega_i_step

    return new_Omega_s, new_Omega_i




def solve_for_steady_state_doppler_broadened_density_matrix(
        Omega_s,         # Signal Rabi Frequency (Related to electric field strength)
        Omega_i,         # Idler Rabi Frequency (Related to electric field strength)
        Omega_I,         # Pump 1 Rabi Frequency (Related to electric field strength)
        Omega_II,        # Pump 2 Rabi Frequency (Related to electric field strength)
        Delta_s,         # Signal detuning
        Delta_i,         # Idler detuning
        Delta_I,         # Pump 1 detuning
        Delta_II,        # Pump 2 detuning
        Delta_a,         # Cavity a detuning
        Delta_b,         # Cavity b detuning
        temp,            # Temperature of Rb gas
        num_velocity_subclasses = 20,
        Hamiltonian_Function = form_Hamiltonian_ab
    ):

    # Set up normal velocity distribution
    sigma_sq = 2 * boltzmann_const * temp / Rb_mass
    maxwell_boltzmann_dist = lambda v: 1 / np.sqrt(np.pi * sigma_sq) * np.exp(-v**2 / sigma_sq)

    # Range of velocities to sample over (2 standard deviations)
    min_v = -2 * np.sqrt(sigma_sq)
    max_v = 2 * np.sqrt(sigma_sq)
    v_vals = np.linspace(min_v, max_v, num_velocity_subclasses)

    rho_sum = None

    for v in v_vals:
        # Frequency dependent doppler shifts
        # We assume all beams travel along the same axis
        # Note that this is an approximation since we take n to be constant
        doppler_shift_s = n * (omega_12 + Delta_s) * v / c # Aligned
        # doppler_shift_i = -n * (omega_34 + Delta_i) * v / c # Anti-aligned

        doppler_shift_I = n * (omega_13 + Delta_I) * v / c # Aligned
        doppler_shift_II = -n * (omega_24 + Delta_I) * v / c # Anti-aligned

        doppler_shift_a = n * (omega_12 + Delta_a) * v / c # Aligned
        doppler_shift_b = -n * (omega_34 + Delta_b) * v / c # Anti-aligned

        # Solve for the steady state
        # NOTE - Hamiltonian_Function is an input argument, not an actual function
        H_total = Hamiltonian_Function(
            Omega_s, Omega_i, Omega_I, Omega_II,
            Delta_s * doppler_shift_s,
            Delta_I * doppler_shift_I,
            Delta_II * doppler_shift_II,
            Delta_a * doppler_shift_a,
            Delta_b * doppler_shift_b
        )
        rho = steadystate(H_total, c_ops, method='direct')

        weight = maxwell_boltzmann_dist(v)

        if rho_sum is None:
            rho_sum = weight * rho
        else:
            rho_sum += weight * rho
        
    # The resulting rho_sum should be approximately normalized,
    # but fix any discrepancy here
    doppler_broadened_rho = rho_sum.unit()

    return doppler_broadened_rho
    

    

def iterative_solve(
        Omega_s,          # Signal Rabi Frequency (Related to electric field strength)
        Omega_i,          # Idler Rabi Frequency (Related to electric field strength)
        Omega_I,          # Pump 1 Rabi Frequency (Related to electric field strength)
        Omega_II,         # Pump 2 Rabi Frequency (Related to electric field strength)
        Delta_s,          # Signal detuning
        Delta_i,          # Idler detuning
        Delta_I,          # Pump 1 detuning
        Delta_II,         # Pump 2 detuning
        Delta_a,          # Cavity a detuning
        Delta_b,          # Cavity b detuning
        temp,             # Temperature
        cell_length,      # Length of vacuum cell (determines number of steps)
        step_size = 0.01, # Distance covered by each iteration
        num_velocity_subclasses = 20,
        Hamiltonian_Function = form_Hamiltonian_ab
    ):

    distance_covered = 0
    while distance_covered < cell_length:
        rho = solve_for_steady_state_doppler_broadened_density_matrix(
            Omega_s, Omega_i, Omega_I, Omega_II,
            Delta_s, Delta_i, Delta_I, Delta_II,
            Delta_a, Delta_b,
            temp,
            num_velocity_subclasses,
            Hamiltonian_Function
        )

        # Update Omega_s and Omega_i via the Maxwell-Bloch equations
        Omega_s, Omega_i = Maxwell_Bloch_step(rho, Omega_s, Omega_i, Delta_s, Delta_i, temp, step_size)

        # Update the distance covered to bring the loop closer to finishing
        distance_covered += step_size
    
    # Obtain the final density matrix with respect to the final values of Omega_s and Omega_i
    rho = solve_for_steady_state_doppler_broadened_density_matrix(
        Omega_s, Omega_i, Omega_I, Omega_II,
        Delta_s, Delta_i, Delta_I, Delta_II,
        Delta_a, Delta_b,
        temp,
        num_velocity_subclasses,
        Hamiltonian_Function
    )

    return rho, Omega_s, Omega_i
