module Predictors

import ..Models: FMModel, predict_instance!

using FactorizationMachines: FMMatrix, FMFloat
using FactorizationMachines.Common: sigmoid
using FactorizationMachines.Tasks: PredictorTask, ClassificationTask, RegressionTask

type FMPredictor{T<:PredictorTask}
    task::T
    model::FMModel
end

"""Instance prediction specialized for classification"""
function predict_instance!(predictor::FMPredictor{ClassificationTask},
                           idx::StridedVector{Int64}, x::StridedVector{FMFloat}, 
                           f_sum::Vector{FMFloat}, sum_sqr::Vector{FMFloat})
    p = predict_instance!(predictor.model, idx, x, f_sum, sum_sqr)
    sigmoid(p)
end

"""Instance prediction specialized for regression"""
function predict_instance!(predictor::FMPredictor{RegressionTask},
                           idx::StridedVector{Int64}, x::StridedVector{FMFloat}, 
                           f_sum::Vector{FMFloat}, sum_sqr::Vector{FMFloat})
    p = predict_instance!(predictor.model, idx, x, f_sum, sum_sqr)
    max(min(p, predictor.task.target_max), predictor.task.target_min)
end

"""Predicts labels for each column of `X`"""
function predict(predictor::FMPredictor, X::FMMatrix)
    result = zeros(X.n)
    predict!(predictor, X, result)
    result
end

"""Predicts labels for each column of `X` and stores the results into `result`"""
function predict!(predictor::FMPredictor, X::FMMatrix, result::Vector{FMFloat})
    fill!(result, .0)
    f_sum = fill(.0, predictor.model.num_factors)
    sum_sqr = fill(.0, predictor.model.num_factors)
    for c in 1:X.n
        X_nzrange = nzrange(X, c)
        idx = sub(X.rowval, X_nzrange)
        x = sub(X.nzval, X_nzrange)
        result[c] = predict_instance!(predictor, idx, x, f_sum, sum_sqr)
    end
end

end
