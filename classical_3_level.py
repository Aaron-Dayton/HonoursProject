import matplotlib.pyplot as plt
import numpy as np
from IPython.display import Image
from qutip import (Qobj, basis, create, destroy, mesolve)

# Global
ℏ = 1

def init_system(Δ=0.01, δ=0.001, Ω_c=4.6, Ω_s=0.3): 
    H_array = -ℏ * np.array([
                    [0,             0,             Ω_s],
                    [0,             δ,             Ω_c],
                    [np.conj(Ω_s),  np.conj(Ω_c),  Δ  ]
                ])

    # Convert to Qobj
    H = Qobj(H_array)

    psi0 = basis(3, 0)
    return H, psi0



def plot_state_expectations():
    H, psi0 = init_system()
    times = np.linspace(0.0, 10.0, 100)

    ket1 = basis(3, 0) # |1>
    ket2 = basis(3, 1) # |2>
    ket3 = basis(3, 2) # |3>

    ρ1 = ket1 * ket1.dag()
    ρ2 = ket2 * ket2.dag()
    ρ3 = ket3 * ket3.dag()
    # Compute results
    result = mesolve(H, psi0, times, [], [ρ1, ρ2, ρ3])

    # Plot
    fig, ax = plt.subplots()
    ax.plot(result.times, result.expect[0])
    ax.plot(result.times, result.expect[1])
    ax.plot(result.times, result.expect[2])
    ax.set_xlabel('Time')
    ax.set_ylabel('Expectation values')
    ax.legend(("Basis state 1", "Basis state 2", "Basis state 3"))
    plt.show()



def plot_position_expectation_over_time(): 
    H, psi0 = init_system()
    times = np.linspace(0.0, 10.0, 100)
 
    ##
    # TODO - This isn't right!
    ##
    # Annihilation operator for 3-state basis
    a = destroy(3)
    # Position operator
    x = (a + a.dag()) / np.sqrt(2)

    # Compute results
    result = mesolve(H, psi0, times, [], [x])

    # Plot
    fig, ax = plt.subplots()
    ax.plot(result.times, result.expect[0])
    ax.set_xlabel('Time')
    ax.set_ylabel('Position expectation value')
    plt.show()



def plot_susceptibility_expectation_over_delta(Δ_end, n=500):
    # Rate of spontaneous emission - Write all other parameters in terms of Γ
    Γ = 1
    Γ13 = 0.5 # One pathway
    Γ23 = 0.5 # Another pathway

    Δ_c = 0 # Perfect EIT transparency
    # Δ_c = 0.1 * Γ # Other behaviour between the two
    # Δ_c = 0.5 * Γ # Raman peak

    Ω_c = 0.1 * Γ
    Ω_s = 10.0**(-5) * Γ

    Δ_vals = np.linspace(-Δ_end, Δ_end, n)
    Γ_times = np.linspace(0.0, 1000.0, 2000) # Just read off the last time step result

    ket1 = basis(3, 0) # |1>
    ket2 = basis(3, 1) # |2>
    ket3 = basis(3, 2) # |3>

    # Dont need?
    # e = 1
    # a_0 = 1
    # d = e * a_0 # Dipole moment (not operator)

    # Dipole operator with dipole approx
    # Lookup oscillator strength and Clebch-Gordon coefficients / Wigner
    # d = \vec{d_13} \cdot \vec{E} e^(-iωt) + \vec{d_13} \cdot \vec{E^*} e^(iωt) |1><3| + h.c.
    # We have \vec{d_13} \cdot \vec{E} = ℏ Ω_s
    # Apply rotating wave approx to get rid of faster freq and we get
    # d = - ℏ Ω_s |1><3| + h.c.
    # P = ε_0 χ E + ε_0 χ^* Ε^* (polarization vector)
    # χ is complex and depends on d (not a physical observable!)
    # So d does not need to be Hermitian and we can simply take
    # d = - ℏ Ω_s |1><3|
    # Without the h.c.
    # This gives us what we want!
    #
    # Although, d = - ℏ Ω_s |3><1| gives a more accurate result and MacRae didn't seem entirely sure.


    d_13 = - ℏ * Ω_s * (ket3 * ket1.dag())
    e_ops = [d_13]

    # Other options
    # d_13 = - ℏ * Ω_s * (ket1 * ket3.dag())
    # d_13 = - ℏ * Ω_s * (ket3 * ket1.dag() + ket3*ket1.dag())

    dissipation_op_13 = Γ13*(ket1 * ket3.dag())
    dissipation_op_23 = Γ23*(ket2 * ket3.dag())
    c_ops = [dissipation_op_13, dissipation_op_23]

    # Other option
    # dissipation_op_13 = Γ13*(ket3 * ket1.dag())
    # dissipation_op_23 = Γ23*(ket3 * ket2.dag())

    real_results = [0] * n
    imag_results = [0] * n

    i = 0
    for Δ in Δ_vals:
        δ = Δ - Δ_c
        # Get system for Δ value
        H, psi0 = init_system(Δ, δ, Ω_c, Ω_s)

        # Compute position expectation value for induced electric dipole
        # TODO - Lookup collapse operators (Lindblad dissipation operators) for incoherent coupling between states
        res = mesolve(H, psi0, Γ_times, c_ops=c_ops, e_ops=e_ops).expect[0][-1]

        # DEBUG for steady state dynamics
        # if i == 50:
        #     temp = mesolve(H, psi0, Γ_times, c_ops, [d_13])
            
        # Compute χ proportionality
        χ = -res**2 * δ / (np.abs(Ω_c)**2 - δ*(Δ + 1j*Γ))
        real_results[i] = χ.real
        imag_results[i] = χ.imag

        i += 1

    
    # Plot
    fig, ax = plt.subplots()
    ax.plot(Δ_vals, real_results)
    ax.plot(Δ_vals, imag_results)
    ax.set_xlabel('Δ')
    ax.set_ylabel('Expectation value of susceptibility χ')
    ax.legend(('Real', 'Imag'))
    plt.title('EIT phenomenon for a 3-level atom in a classical EM field')

    # ax.plot(temp.times, temp.expect[0])
    plt.show()