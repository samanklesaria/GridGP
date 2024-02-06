using LinearAlgebra
import Random: rand!, default_rng
using Infiltrator

function lanczos(A, v1, m)
    v = copy(v1)
    T = eltype(v)
    V = Matrix{T}(undef, length(v), m)
    V[:, 1] = v
    w = Vector{T}(undef, length(v1))
    As = Vector{T}(undef, m)
    Bs = Vector{T}(undef, m-1)
    β = 0.0
    for i in 1:m
        mul!(w, A, v)
        α = dot(w, v)
        As[i] = α
        w[:] .-= α .* v
        if i > 1
            w -= β * V[:, i-1]
        end
        β = norm(w)
        if iszero(β)
            return V[:, 1:i], SymTridiagonal(As[1:i], Bs[1:i-1])
        end
        if i == m
            return V, SymTridiagonal(As, Bs)
        end
        Bs[i] = β
        v[:] .= w ./ β
        V[:, i+1] = v
    end
end

pos_log(x) = log(max(zero(x), x))

function slq(A, δ, m, nv)
    rng = default_rng()
    n = size(A, 1)
    u = Array{Float64}(undef, n)
    Γ = 0.0
    V = Matrix{Float64}(undef, 0,0)
    T = SymTridiagonal(Float64[], Float64[])
    for _ in 1:nv
        rand!(rng, u, (-1, 1))
        normalize!(u)
        V, T = lanczos(A, u, m)
        F = eigen(T)
        Γ += sum(pos_log.(F.values) .* F.vectors[1, :] .^ 2)
    end
    α = V * (T \ (V' * δ))
    (n/nv) * Γ, α
end

# Blah. Need to think about gradients more.
# Also: do a reverse rule. 


# ALSO: would be good to implement the exact log det
# in places where it's tractable

# TODO: make Laplace approximation use Lanczos factorization
# Understand what gradients for stationary kernels look like

function ChainRulesCore.frule((_, ΔA, _, _, _), ::typeof(slq), A, δ, m, nv)
  rng = default_rng()
  n = size(A, 1)
  u = Array{Float64}(undef, n)
  Γ = 0.0
  g = 0.0
  V = Matrix{Float64}(undef, 0,0)
  T = SymTridiagonal(Float64[], Float64[])
  e1 = sparsevec(1:1, [1.], m)
  for _ in 1:nv
      rand!(rng, u, (-1, 1))
      normalize!(u)
      Q, T = lanczos(A, u, m)
      g += dot(Q * (T \ e1), ΔA * z)
      F = eigen(T)
      Γ += sum(pos_log.(F.values) .* F.vectors[1, :] .^ 2)
  end
  α = V * (T \ (V' * δ))
  ((n/nv) * Γ, α), (g / nv, - V * (T \ (V' * ΔA * α)))
end
