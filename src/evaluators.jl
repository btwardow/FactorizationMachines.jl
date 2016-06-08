module Evaluators
export Evaluator

using FactorizationMachines: FMMatrix, FMFloat, sqerr, z1loss
using ..Tasks: PredictorTask
using ..Predictors: FMPredictor, predict!

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
    err = [sqerr(predictions[i], y[i]) for i in 1:length(y)]
    evaluator.stat(err .* err)
end

immutable ZeroOneEvaluator <: Evaluator
    stat::Function

    ZeroOneEvaluator(; stat::Function = x -> mean(x)) = new(stat)
end
const z1 = ZeroOneEvaluator

function evaluate{T<:PredictorTask}(evaluator::ZeroOneEvaluator, predictor::FMPredictor{T}, 
        X::FMMatrix, y::Array{FMFloat})
    predictions = zeros(X.n)
    evaluate!(evaluator, predictor, X, y, predictions)
end

function evaluate!{T<:PredictorTask}(evaluator::ZeroOneEvaluator, predictor::FMPredictor{T}, 
        X::FMMatrix, y::Array{FMFloat}, predictions::Array{FMFloat})
    @time predict!(predictor, X, predictions)
    err = [z1loss(predictions[i], y[i]) for i in 1:length(y)]
    evaluator.stat(err)
end

end
