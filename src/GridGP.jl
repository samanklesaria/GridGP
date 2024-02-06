module GridGP
export GridApprox, CG_GP

using ToeplitzMatrices, AbstractGPs, Statistics, FFTW,
FillArrays, IterativeSolvers, KernelFunctions,
SparseArrays, SparseArrays, Interpolations, ChainRulesCore
import LinearMaps: LinearMap
import Distributions

include("slq.jl")

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

struct GridApprox{T,R,L,I,D} <: Kernel
  k::T
  g::Grid{R}
  k_uu::L
  itp::I
  dims::D
end

function GridApprox(k, g)
  dims = length.(g)
  itp = interpolate(Zeros(dims), BSpline(Cubic()))
  k_uu = kron(LinearMap.(kernelmatrix.(k.kernels, g))...)
  GridApprox(k, g, k_uu, Interpolations.itpinfo(itp), dims)
end

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

function to_sparse(w, dim)
  ixs = clamp.(w.istart : (w.istart + length(w.weights) - 1), 1, dim)
  sparsevec(ixs, collect(w.weights), dim)
end

struct Lanczos{F}
  m::Int
  f::F
end

function AbstractGPs.posterior(fx::FiniteGP{Lanczos}, y::AbstractVector)
    k = fx.f.f.kernel
    K = kernelmatrix(k, fx.x) + fx.Σy
    δ = y - mean(fx)
    n = norm(δ)
    V, T = lanczos(K, δ / n, fx.f.m)
    e1 = sparsevec(1:1, [1.], size(T, 1))
    α = n * (V * (T \ e1))
    W = weights(k, xs)
    R = k.k_uu * (W * V)
    R2 = T \  R
    PosteriorGP(fx.f, (α=α, R=R, R2=R2))
end

function Statistics.var(g::PosteriorGP{Lanczos}, xs::AbstractVector)
  W = weights(g.f.kernel, xs)
  l = RowVecs(W * R) # a x k
  r = ColVecs(R2 * W') # k x a
  var(g.prior.f, xs) .- dot.(l, r)
end

function Random.rand(rng::AbstractRNG, fx::FiniteGP{PosteriorGP{Lanczos}} , N::Int)
  k = fx.f.prior.f.kernel
  k_uu = k.k_uu 
  m = fx.f.prior.f.m
  A = k_uu - LinearMap(fx.f.data.R) * LinearMap(fx.f.data.R2)
  e1 = sparsevec(1:1, [1.], size(A, 1))
  V, T = lanczos(A, e1, m)
  S = V * cholesky(T).L
  W = weights(k, fx.x)
  mean(fx) + W * (S * randn(rng, m, N))
end



# TODO: how does reverse diff with Toeplitz and Kronecker structure
# work?

## ---


function halfkern(k::GridApprox, xs::AbstractVector)
  W = weights(k, xs)
  k.k_uu * W1'
end

struct CG_GP{F} <: AbstractGPs.AbstractGP
    f::F
end

Statistics.mean(g::CG_GP, x::AbstractVector) = mean(g.f, x)
Statistics.cov(g::CG_GP, x::AbstractVector, y::AbstractVector) = cov(g.f, x, y)
Statistics.var(g::CG_GP, x::AbstractVector) = var(g.f, x)

function AbstractGPs.posterior(fx::AbstractGPs.FiniteGP{CG_GP}, y::AbstractVector{<:Real})
    k = fx.f.f.kernel
    K = kernelmatrix(k, fx.x) + fx.Σy
    δ = y - mean(fx)
    α = cg(K, δ)
    h = halfkern(k, fx.x)
    AbstractGPs.PosteriorGP(fx.f, (α=α, K=K, δ=δ, h=h))
end

function Statistics.mean(g::PosteriorGP{CG_GP}, xs::AbstractVector)
  mean(f.prior, x) + weights(g.prior.f.kernel, xs) * g.data.h * g.data.α
end


# TODO: we also should choose the number of points dynamically.
# Wait until the expectation stabilizes. 

# When evaluating the mean, we want K_*x (W_xu K_uu W_ux)^{-1} y
# This much is calculated correctly. 

# TODO: we want to cache

# When doing the variance calculation, though, we need access to k_uu by itself. 


function slq_logdet(A)
  slq(pos_log, A, clamp(trunc(Int, size(A, 1) / 4), 1, 16), 16)
end

function Distributions.logpdf(fx::AbstractGPs.FiniteGP{<:CG_GP}, y::AbstractVecOrMat{<:Real})
  post = posterior(fx, y)
  quadform = post.data.α' * post.data.δ
  ld = slq_logdet(post.data.C)
  -0.5 * ld -(length(y)/2) * log(2 * pi) - 0.5 * quadform
end

function ChainRulesCore.frule((_, ΔA), ::typeof(slq_logdet), A)
  nv = 16
  m = clamp(trunc(Int, size(A, 1) / 4), 1, 16)
  rng = default_rng()
  n = size(A, 1)
  u = Array{Float64}(undef, n)
  Γ = 0.0
  g = 0.0
  e1 = zeros(n)
  e1[1] = 1
  for i in 1:nv
      rand!(rng, u, (-1, 1))
      normalize!(u)
      Q, T = lanczos(A, u, m)
      g += dot(Q * (T \ e1), ΔA * z)
      F = eigen(T)
      Γ += sum(pos_log.(F.values) .* F.vectors[1, :] .^ 2)
  end
  (n/nv) * Γ, g / nv
end

end # module GridGP
