import matplotlib.pyplot as plt
import numpy as np
from IPython.display import Image
from qutip import (Qobj, about, basis, coherent, coherent_dm, create, destroy,
                   expect, fock, fock_dm, mesolve, qeye, sigmax, sigmay,
                   sigmaz, tensor, thermal_dm, anim_matrix_histogram,
                   anim_fock_distribution)

####
# SOURCE FOR THE FOLLOWING CODE:
# https://qutip.org/docs/4.0.2/guide/dynamics/dynamics-master.html
####

def schrodinger_mesolve():
    # Form hamiltonian and initial wavefunction which will have their expectation values taken
    # with respect to σ_z and σ_y at times between 1 and 10 on the timescale of the system.
    H = 2 * np.pi * 0.1 * sigmax()
    psi0 = basis(2, 0)
    times = np.linspace(0.0, 10.0, 100)

    # Compute results
    result = mesolve(H, psi0, times, [], [sigmaz(), sigmay()])

    # Plot
    fig, ax = plt.subplots()
    ax.plot(result.times, result.expect[0])
    ax.plot(result.times, result.expect[1])
    ax.set_xlabel('Time')
    ax.set_ylabel('Expectation values')
    ax.legend(("Sigma-Z", "Sigma-Y"))
    plt.show()

def master_mesolve():
    # Form hamiltonian and initial wavefunction which will have their expectation values taken
    # with respect to σ_z and σ_y at times between 1 and 10 on the timescale of the system.
    H = 2 * np.pi * 0.1 * sigmax()
    psi0 = basis(2, 0)
    times = np.linspace(0.0, 10.0, 100)

    # Get result with dissipation operator σ_x with dissipative rate 0.05
    result = mesolve(H, psi0, times, [np.sqrt(0.05) * sigmax()], [sigmaz(), sigmay()])

    # Plot
    fig, ax = plt.subplots()
    ax.plot(times, result.expect[0])
    ax.plot(times, result.expect[1])
    ax.set_xlabel('Time')
    ax.set_ylabel('Expectation values')
    ax.legend(("Sigma-Z", "Sigma-Y"))
    plt.show()

# Two level system with a dissipative cavity
def two_level():
    # Two level system
    psi0 = tensor(fock(2,0), fock(10, 5))
    a = tensor(qeye(2), destroy(10)) # Lowering operator (I think)
    sm = tensor(destroy(2), qeye(10))
    H = 2 * np.pi * a.dag() * a + 2 * np.pi * sm.dag() * sm + 2 * np.pi * 0.25 * (sm * a.dag() + sm.dag() * a)
    times = np.linspace(0.0, 10.0, 200)

    # Get result with dissipative operators (raising and lowering operators)
    result = mesolve(H, psi0, times, [np.sqrt(0.1)*a], [a.dag()*a, sm.dag()*sm])
    plt.figure()
    plt.plot(times, result.expect[0])
    plt.plot(times, result.expect[1])
    plt.xlabel('Time')
    plt.ylabel('Expectation values')
    plt.legend(("cavity photon number", "atom excitation probability"))
    plt.show()