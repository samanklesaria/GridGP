module GridGP

using ToeplitzMatrices, AbstractGPs, Statistics, LinearMaps, FFTW,
FillArrays, IterativeSolvers, KernelFunctions, Kronecker,
SparseArrays, Unzip

using SparseArrays

struct Stationary{T}
  k::T
end

function KernelFunctions.kernelmatrix(k::Stationary, x::StepRange)
  SymmetricToeplitz(k.k.(x[1], x))
end

function KernelFunctions.kernelmatrix(k::Stationary, x::StepRange,
  y::StepRange)
  if x === y
    kernelmatrix(k, x,y)
  elseif x.step == y.step
    Toeplitz(k.k.(x, y[1]), k.k.(x[1], y))
  else
    kernelmatrix(k.k, x, y)
  end
end

struct GridApprox{T,G,L}
  k::T
  g::G
  k_uu::L
end

GridApprox(k, g) = GridApprix(k, g, LinearMap(kernelmatrix(k, g)))

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

function grid_ix(pts::StepRange, pt::Number)
    Int(1 + ceil((pt - pts.start) / pts.step))
end

function weights(pts, xs)
    ixs, vals = unzip(weight.(Ref(pts), xs))
    i = 1:length(xs)
    sparse(vec([i i]), collect(Iterators.flatten(ixs)),
      collect(Iterators.flatten(vals)))
end

function weight(pts::StepRange, x::Float64)
    ix = grid_ix(pts, x)
    if pts[ix] == x
        return ([ix], [1.])
    end
    closest = pts[[ix - 1, ix]]
    if closest == 1
        ([closest], [1.])
    else
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

# function logpdf(fx::AbstractGP.FiniteGP{<:CG_GP}, y::AbstractVecOrMat{<:Real})
#   post = posterior(fx, y)
#   quadform = post.data.α' * post.data.δ
#   logdet = sum(log.(eigvals(post.data.K) .+ fx.Σy.diag))
#   k = length(y)
#   -0.5 * logdet -(k/2) * log(2 * pi) - 0.5 * quadform
# end

end # module GridGP
