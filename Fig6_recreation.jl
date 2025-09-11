using DSP
using LinearAlgebra
using Plots

B = 1
γ = 1
Ω = 3
θ(τ) = (τ >= 0) ? 1.0 : 0.0
ψ(τ) = B * exp(-γ * τ) * sin(Ω*τ/2) * θ(τ)
Ψ(τ) = ψ(τ) + ψ(-τ)

L = 6
V = 1
Π(τ) = ( (τ >= -L/V) && (τ <= L/V) ) ? (V/(2*L)) : 0

τ = -10:0.01:10
Ψτ = Ψ.(τ)
Ψτ = Ψτ ./ norm(Ψτ)
Πτ = Π.(τ)
Πτ = Πτ ./ norm(Πτ)


conv_result = conv(Ψτ, Πτ)
conv_result = conv_result ./ norm(conv_result)

dτ = step(τ)
conv_τ = (first(τ) + first(τ)) : dτ : (last(τ) + last(τ))

plot(τ, Πτ, label="Π(τ)", title="Eq. 9 Plot")
plot!(τ, Ψτ, label="Ψ(τ)")
plot!(conv_τ, conv_result, label="Ψ * Π")