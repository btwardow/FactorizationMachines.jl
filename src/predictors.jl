module Predictors

using FactorizationMachines: FMMatrix, FMFloat, sigmoid
import ..Models: FMModel, predict_instance!
using ..Tasks: PredictorTask, ClassificationTask, RegressionTask

type FMPredictor{T<:PredictorTask}
    task::T
    model::FMModel
end

"""Instance prediction specialized for classification"""
function predict_instance!(predictor::FMPredictor{ClassificationTask},
                           idx::Array{Int64,1}, x::Array{FMFloat,1}, 
                           f_sum::Array{FMFloat}, sum_sqr::Array{FMFloat})
    p = predict_instance!(predictor.model, idx, x, f_sum, sum_sqr)
    sigmoid(p)
end

"""Instance prediction specialized for regression"""
function predict_instance!(predictor::FMPredictor{RegressionTask},
                           idx::Array{Int64,1}, x::Array{FMFloat,1}, 
                           f_sum::Array{FMFloat}, sum_sqr::Array{FMFloat})
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
function predict!(predictor::FMPredictor, X::FMMatrix, result::Array{FMFloat})
    fill!(result, .0)
    f_sum = fill(.0, predictor.model.num_factors)
    sum_sqr = fill(.0, predictor.model.num_factors)
    for c in 1:X.n
        idx = X.rowval[X.colptr[c] : (X.colptr[c+1]-1)]
        x = X.nzval[X.colptr[c] : (X.colptr[c+1]-1)]
        result[c] = predict_instance!(predictor, idx, x, f_sum, sum_sqr)
    end
end

end
