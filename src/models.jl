module Models
export ModelParams

using FactorizationMachines: FMMatrix, FMFloat

abstract ModelParams

immutable GaussianModelParams <: ModelParams
    k0::Bool
    k1::Bool
    num_factors::Int

    mean::Float64
    stddev::Float64

    GaussianModelParams(; k0 = true, k1 = true, num_factors = 8, mean = .0, stddev = .01) = new(k0, k1, num_factors, mean, stddev)
end
const gauss = GaussianModelParams

function init(params::GaussianModelParams, X::FMMatrix, y::Vector{FMFloat})
    # initialization
    num_attributes, num_samples = size(X)
    # sanity check
    assert(length(y) == num_samples)

    # create initial model
    w0 = .0
    w = zeros(num_attributes)
    V = randn(params.num_factors, num_attributes) .* params.stddev + params.mean

    # new model
    model = FMModel(params.k0, params.k1, w0, w, V, params.num_factors)
end

type FMModel
    k0::Bool
    k1::Bool

    w0::FMFloat
    w::Vector{FMFloat}
    V::Matrix{FMFloat}

    num_factors::Int
end

function predict_instance!(model::FMModel, 
                           idx::StridedVector{Int64}, x::StridedVector{FMFloat}, 
                           f_sum::Vector{FMFloat}, sum_sqr::Vector{FMFloat})
    fill!(f_sum, .0)
    fill!(sum_sqr, .0)
    result = zero(FMFloat)
    if model.k0
        result += model.w0
    end
    if model.k1
        for i in 1:length(idx)
            result += model.w[idx[i]] * x[i]
        end
    end
    for f in 1:model.num_factors
        for i in 1:length(idx)
            d = model.V[f,idx[i]] * x[i]
            f_sum[f] += d
            sum_sqr[f] += d * d
        end
        result += 0.5 * (f_sum[f] * f_sum[f] - sum_sqr[f])
    end
    result
end

end
