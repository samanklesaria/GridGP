module GridGP
export GridApprox, CG_GP, Grid

using ToeplitzMatrices, AbstractGPs, Statistics, LinearMaps, FFTW,
FillArrays, IterativeSolvers, KernelFunctions, Kronecker,
SparseArrays, Unzip, SparseArrays

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

struct GridApprox{T,G,L} <: Kernel
  k::T
  g::G
  k_uu::L
end

struct Grid{T}
  dims::T
end
  
function KernelFunctions.kernelmatrix(k::KernelTensorProduct,
    x::Grid)
  kron(kernelmatrix.(k.kernels, x.dims)...)
end

GridApprox(k, g) = GridApprox(k, g, LinearMap(kernelmatrix(k, g)))

function KernelFunctions.kernelmatrix(g::GridApprox, x::AbstractVector)
  W = weights(g.g, x)
  W * g.k_uu * W'
end

function KernelFunctions.kernelmatrix(g::GridApprox,
    x::AbstractVector, y::AbstractVector)
  W1 = weights(g.g, x)
  W2 = weights(g.g, y)
  W1 * g.k_uu * W2'
end

function grid_ix(pts::AbstractRange, pt::T) where {T <: Number}
    Int(1 + ceil((pt - pts[1]) / T(step(pts))))
end

function weights(pts::AbstractRange, xs::AbstractVector{T}) where {T <: Number}
  ivec = Int[]
  jvec = Int[]
  valvec = Float64[]
  ix_data = weight.(Ref(pts), xs)
  for (i, (js, vals)) in enumerate(ix_data)
      append!(ivec, fill(i, length(js)))
      append!(jvec, js)
      append!(valvec, vals)
  end
  sparse(ivec, jvec, valvec, length(xs), length(pts))
end

function weight(pts::AbstractRange, x::Number)
    ix = grid_ix(pts, x)
    if pts[ix] == x
        return ([ix], [1.])
    elseif ix == 1
        return ([1], [1.])
    else
        closest = pts[[ix - 1, ix]]
        l = closest[2] - closest[1]
        vals = abs.(x .- closest) ./ l
        ([ix, ix-1], vals)
    end
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

# TODO: logpdf for cg-gps

# function logpdf(fx::AbstractGP.FiniteGP{<:CG_GP}, y::AbstractVecOrMat{<:Real})
#   post = posterior(fx, y)
#   quadform = post.data.α' * post.data.δ
#   logdet = sum(log.(eigvals(post.data.K) .+ fx.Σy.diag))
#   k = length(y)
#   -0.5 * logdet -(k/2) * log(2 * pi) - 0.5 * quadform
# end

end # module GridGP
