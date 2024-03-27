module GridGP
using AbstractGPs: PosteriorGP
export GridApprox, CG_GP

using ToeplitzMatrices, AbstractGPs, Statistics, FFTW,
FillArrays, IterativeSolvers, KernelFunctions, LinearAlgebra,
SparseArrays, SparseArrays, Interpolations, ChainRulesCore
import LinearMaps: LinearMap
import Distributions

# Simple Kernels are stationary, and should result in Toeplitz matrices.

KernelFunctions.kernelmatrix(k::KernelFunctions.SimpleKernel, x::AbstractRange) = 
  SymmetricToeplitz(k.(x[1], x))

function KernelFunctions.kernelmatrix(k::KernelFunctions.SimpleKernel,
    x::AbstractRange, y::AbstractRange)
  if x === y
    SymmetricToeplitz(k.k.(x[1], x))
  elseif step(x) == step(y)
    Toeplitz(k.(x, y[1]), k.k.(x[1], y))
  else
    k.(x, y')
  end
end

"""
A `GridApprox` overloads `kernelmatrix` to interpolate
on a grid of inducing points. The type paramters are as follows:
- `T` is the type of the underlying kernel. It must be a kernel tensor
product of some kind, with each kernel in the product handling a
different dimension in the grid of inducing points.
- `G` is the grid type (a vector of some kind of `AbstractRange`).
- `L` is the type of the Kronecker structured inducing
point covariance matrix.
- `I` is an interoplation object.
"""
struct GridApprox{T,R,L,I,D} <: Kernel
  k::T
  g::R
  k_uu::L
  itp::I
  dims::Vector{Int}
end

"""
Construct a `GridApprox`. 
Each kernel in `k` acts on a different dimension of the grid. 
Grid points in dimension `i are spaced according to the `AbstractRange` `g[i]`.
"""
function GridApprox(k::KernelTensorProduct, g::Vector{T}) where {T <: AbstractRange}
  dims = length.(g)
  itp = interpolate(Zeros(dims), BSpline(Cubic()))
  k_uu = kronecker(kernelmatrix.(k.kernels, g)...)
  GridApprox(k, g, k_uu, Interpolations.itpinfo(itp), dims)
end

"""
Construct a sparse matrix that interpolates among the inducing points in `GridApprox`
to capture the covariance among `xs`.
"""
function weights(k::GridApprox, xs::AbstractVector)
  inits = [r[1] for r in k.g]
  steps = eltype(xs[1]).(step.(k.g))
  rs = Int[]
  cs = Int[]
  vals = Float64[]
  n = 0 
  for (i, x) in enumerate(xs)
    x0 = Tuple((x .- inits) ./ steps .+ 1)
    wis = Interpolations.weightedindexes((Interpolations.value_weights,),
      k.itp..., x0)
    v = kron(to_sparse.(wis, k.dims)...)
    n = length(v)
    append!(cs, v.nzind)
    append!(rs, fill(i, length(v.nzind)))
    append!(vals, v.nzval)
  end
  sparse(rs, cs, vals, length(xs), n)
end

function KernelFunctions.kernelmatrix(k::GridApprox, xs::AbstractVector)
  W = weights(k, xs)
  W * k.k_uu * W'
end

"Turn a WeightedIndices into a size `dim` sparse vector of indices"
function to_sparse(w, dim)
  ixs = clamp.(w.istart : (w.istart + length(w.weights) - 1), 1, dim)
  sparsevec(ixs, collect(w.weights), dim)
end

"""
Uses the algorithm in (Gardner and Pleiss. GPyTorch: Blackbox Matrix-Matrix Gaussian Process
Inference with GPU Acceleration. NeurIPS 2021) to find GP likelihood.
"""
struct Gardner
    t::Int # Number of batches
    p::Int # Steps of batched CG
end

struct LOVE
    α::Vector{Float64}
    x::Vector{Float64}
    δ::Vector{Float64}
    R::Matrix{Float64}
    R2::Matrix{Float64}
end

function AbstractGPs.posterior(strat::Gardner, fx::FiniteGP, y::AbstractVector)
    kern = fx.f.kernel
    m = size(kern.k_uu, 1)
    K = LinearMap(kernelmatrix(kern, fx.x)) + fx.Σy
    δ = y - mean(fx)
    W = weights(kern, fx.x)
    b = (W * kern.k_uu * Ones(m)) / m
    U, Ts, Qs = mbcg(K, [δ; b], Val(true))
    R = kern.k_uu * (W * Qs[2])
    R2 = Ts[2] \ R'
    AbstractGPs.PosteriorGP(fx.f, LOVE(U[:, 1], fx.x, δ, R, R2))
end

const LovePosterior = PosteriorGP{UnionAll, Love}

function Statistics.var(g::LovePosterior, xs::AbstractVector)
  W = weights(g.f.kernel, xs)
  l = RowVecs(W * g.data.R) # a x k
  r = ColVecs(g.data.R2 * W') # k x a
  var(g.prior.f, xs) .- dot.(l, r)
end

"Lazy application of `vec`"
struct Vec{K}
  k::K
end

"""
If ``a_i`` and ``b_i`` are the columns of `A` and `B` respectively, this
is a lazy representation of ``vec\left(\sum_i a_i b^_i^T\right)``.
"""
struct OuterSum
  A::Matrix{Float64}
  B::Matrix{Float64}
end

Base.*(v::Vec, s::OuterSum) = dot(vec(v.k * s.A), vec(s.B))

function approx_lml(strat::Gardner, fx::AbstractGPs.FiniteGP, y::AbstractVector)
    kern = fx.f.kernel
    K = kernelmatrix(kern, fx.x) + fx.Σy
    δ = y - mean(fx)
    return logp_with(strat, δ, Vec(K))
end

function logp_with(strat, δ, K)
    n = length(δ)
    z = randn(n, strat.t)
    u, Ts = mbcg(K.k, [δ; z])
    ld = mean([
        begin
            F = eigen(T)
            sum(F.vectors[1, :].^2 .* log.(F.values))
        end for T in Ts])
    -n / 2 * log(2 * pi) - 0.5 * (ld + dot(δ, u[:, 1]))
end

function ChainRulesCore.frule((_, _, _, ΔK), ::typeof(logp_with), strat::Gardner, δ, K)
    n = length(δ)
    z = randn(n, strat.t)
    u, Ts = mbcg(K.k, [δ; z])
    kxx_inv_δ = u[:, 1]
    tangent = OuterSum([kxx_inv_δ; z / strat.t], [kxx_inv_δ; u[:, 2:end]])
    Δresult = ΔK * tangent 
    ld = mean([
        begin
            F = eigen(T)
            sum(F.vectors[1, :].^2 .* log.(F.values))
        end for T in Ts])
    result = -n / 2 * log(2 * pi) - 0.5 * (ld + dot(δ, u[:, 1]))
    return (result, Δresult)
end

function ChainRulesCore.rrule(:typeof(logp_with), strat::Gardner, δ, K)
    n = length(δ)
    z = randn(n, strat.t)
    u, Ts = mbcg(K, [δ; z])
    kxx_inv_δ = u[:, 1]
    ld = mean(Float64[
        begin
            F = eigen(T)
            sum(F.vectors[1, :].^2 .* log.(F.values))
        end for T in Ts])
    result = -n / 2 * log(2 * pi) - 0.5 * (ld + dot(δ, u[:, 1]))
    function pullback(ΔL)
       tangent = OuterSum([kxx_inv_δ; z / strat.t], [kxx_inv_δ; u[:, 2:end]])
       (NoTangent(), NoTangent(), NoTangent(), tangent)
    end
    return result, pullback
end

function Random.rand(rng::AbstractRNG, fx::FiniteGP{LovePosterior} , N::Int)
  kern = fx.f.prior.kernel
  k_uu = kern.k_uu 
  A = k_uu - LinearMap(fx.f.data.R) * LinearMap(fx.f.data.R2)
  e1 = sparsevec(1:1, [1.], size(A, 1))
  V, T = lanczos(A, e1)
  S = V * cholesky(T).L
  W = weights(kern, fx.x)
  mean(fx) + W * (S * randn(rng, size(S, 1), N))
end


end # module GridGP
