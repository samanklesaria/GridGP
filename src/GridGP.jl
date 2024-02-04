module GridGP
export GridApprox, CG_GP, Grid

using ToeplitzMatrices, AbstractGPs, Statistics, FFTW,
FillArrays, IterativeSolvers, KernelFunctions,
SparseArrays, SparseArrays, Interpolations
import LinearMaps: LinearMap
import Distributions

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

struct Grid{T}
  ranges::T
end

struct GridApprox{T,R,L} <: Kernel
  k::T
  g::Grid{R}
  k_uu::L
end

GridApprox(k, g) = GridApprox(k, g, kernelmatrix(k, g))

function KernelFunctions.kernelmatrix(k::KernelTensorProduct,
    x::Grid)
  kron(LinearMap.(kernelmatrix.(k.kernels, x.ranges))...)
end

function weights(itp, ranges, dims, xs)
  inits = [r[1] for r in ranges]
  steps = eltype(xs[1]).(step.(ranges))
  rs = Int[]
  cs = Int[]
  vals = Float64[]
  n = 0 
  for (i, x) in enumerate(xs)
    x0 = Tuple((x .- inits) ./ steps .+ 1)
    wis = Interpolations.weightedindexes((Interpolations.value_weights,),
      Interpolations.itpinfo(itp)..., x0)
    v = kron(to_sparse.(wis, dims)...)
    n = length(v)
    append!(cs, v.nzind)
    append!(rs, fill(i, length(v.nzind)))
    append!(vals, v.nzval)
  end
  sparse(rs, cs, vals, length(xs), n)
end

function KernelFunctions.kernelmatrix(k::GridApprox, xs::AbstractVector)
  dims = length.(k.g.ranges)
  itp = interpolate(Zeros(dims), BSpline(Cubic()))
  W = weights(itp, k.g.ranges, dims, xs)
  W * k.k_uu * W'
end

function KernelFunctions.kernelmatrix(k::GridApprox,
  xs::AbstractVector, ys::AbstractVector)
  dims = length.(k.g.ranges)
  itp = interpolate(Zeros(dims), BSpline(Cubic()))
  W1 = weights(itp, k.g.ranges, dims, ys)
  W2 = weights(itp, k.g.ranges, dims, xs)
  W2 * k.k_uu * W1'
end

function to_sparse(w, dim)
  ixs = clamp.(w.istart : (w.istart + length(w.weights) - 1), 1, dim)
  sparsevec(ixs, collect(w.weights), dim)
end
 
struct CG_GP{F} <: AbstractGPs.AbstractGP
    f::F
end

Statistics.mean(g::CG_GP, x::AbstractVector) = mean(g.f, x)
Statistics.cov(g::CG_GP, x::AbstractVector, y::AbstractVector) = cov(g.f, x, y)
Statistics.var(g::CG_GP, x::AbstractVector) = var(g.f, x)

function AbstractGPs.posterior(fx::AbstractGPs.FiniteGP{<:CG_GP}, y::AbstractVector{<:Real})
    k = fx.f.f.kernel
    K = kernelmatrix(k, fx.x) + fx.Σy
    δ = y - mean(fx)
    α = cg(K, δ)
    AbstractGPs.PosteriorGP(fx.f, (α=α, C=K, x=fx.x, δ=δ))
end


function Distributions.logpdf(fx::AbstractGPs.FiniteGP{<:CG_GP}, y::AbstractVecOrMat{<:Real})
  k = length(y)
  post = posterior(fx, y)
  quadform = post.data.α' * post.data.δ
  A = post.data.C
  m = max(1, trunc(Int, k / 3))
  X0 = rand(eltype(A), size(A, 1), m)

  highest = lobpcg(A, true, X0, m).λ
  lowest = lobpcg(A, false, X0, m).λ
  mid_guess = log((highest[end] + lowest[end]) / 2)
  logdet = sum(log.(highest)) + sum(log.(lowest)) + mid_guess * (k - (2 * m))
  -0.5 * logdet -(k/2) * log(2 * pi) - 0.5 * quadform
end

end # module GridGP
