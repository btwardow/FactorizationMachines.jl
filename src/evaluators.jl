module Evaluators
export Evaluator

using FactorizationMachines: FMMatrix, FMFloat
using FactorizationMachines.Common
using FactorizationMachines.Predictors: FMPredictor, predict!
using FactorizationMachines.Tasks: PredictorTask

abstract Evaluator

immutable SquaredErrorEvaluator <: Evaluator
    stat::Function

    SquaredErrorEvaluator(; stat::Function = x -> sqrt(mean(x))) = new(stat)
end
const rmse = SquaredErrorEvaluator

function evaluate{T<:PredictorTask}(evaluator::SquaredErrorEvaluator, predictor::FMPredictor{T}, 
        X::FMMatrix, y::Array{FMFloat})
    predictions = zeros(X.n)
    evaluate!(evaluator, predictor, X, y, predictions)
end

function evaluate!{T<:PredictorTask}(evaluator::SquaredErrorEvaluator, predictor::FMPredictor{T}, 
        X::FMMatrix, y::Array{FMFloat}, predictions::Array{FMFloat})
    @time predict!(predictor, X, predictions)
    err = [Common.sqerr(predictions[i], y[i]) for i in 1:length(y)]
    evaluator.stat(err .* err)
end

immutable HeavisideEvaluator <: Evaluator
    stat::Function

    HeavisideEvaluator(; stat::Function = x -> mean(x)) = new(stat)
end
const heaviside = HeavisideEvaluator

function evaluate{T<:PredictorTask}(evaluator::HeavisideEvaluator, predictor::FMPredictor{T}, 
        X::FMMatrix, y::Array{FMFloat})
    predictions = zeros(X.n)
    evaluate!(evaluator, predictor, X, y, predictions)
end

function evaluate!{T<:PredictorTask}(evaluator::HeavisideEvaluator, predictor::FMPredictor{T}, 
        X::FMMatrix, y::Array{FMFloat}, predictions::Array{FMFloat})
    @time predict!(predictor, X, predictions)
    err = [Common.heaviside(predictions[i], y[i]) for i in 1:length(y)]
    evaluator.stat(err)
end

end
