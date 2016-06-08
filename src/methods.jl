module Methods
export MethodParams

using FactorizationMachines: FMMatrix, FMFloat, FMInt
using ..Tasks: PredictorTask, loss_deriv
using ..Models: FMModel
using ..Predictors: FMPredictor, predict_instance!
using ..Evaluators: Evaluator, evaluate!

abstract MethodParams

immutable SGDMethod <: MethodParams
  alpha::FMFloat
  num_epochs::UInt

  # regularization
  reg0::FMFloat
  regw::FMFloat
  regv::FMFloat

  SGDMethod(; alpha::FMFloat = 0.01, num_epochs::UInt = UInt(100),
              reg0::FMFloat = .0, regw::FMFloat = .0, regv::FMFloat = .0) = new(alpha, num_epochs, reg0, regw, regv)
end
const sgd = SGDMethod

function sgd_train!{T<:PredictorTask}(sgd::SGDMethod, evaluator::Evaluator, predictor::FMPredictor{T}, X::FMMatrix, y::Array{FMFloat})
   info("Learning Factorization Machines with gradient descent...")
   for epoch in 1:sgd.num_epochs
        #info("[SGD - Epoch $epoch] Start...")
        @time sgd_epoch!(sgd, evaluator, predictor, X, y, epoch, sgd.alpha)
        #info("[SGD - Epoch $epoch] End.")
   end
end

function sgd_epoch!{T<:PredictorTask}(sgd::SGDMethod, evaluator::Evaluator, predictor::FMPredictor{T}, X::FMMatrix, y::Array{FMFloat}, epoch::Integer, alpha::FMFloat)
    predictions = zeros(length(y))
    p = .0
    f_sum = fill(.0, predictor.model.num_factors)
    sum_sqr = fill(.0, predictor.model.num_factors)
    mult = .0

    for c in 1:X.n
        idx = X.rowval[X.colptr[c] : (X.colptr[c+1]-1)]
        x = X.nzval[X.colptr[c] : (X.colptr[c+1]-1)]
        #info("DEBUG: processing $c")
        p = predict_instance!(predictor, idx, x, f_sum, sum_sqr)
        #info("DEBUG: prediction - p: $p, f_sum: $f_sum, sum_sqr: $sum_sqr")
        mult = loss_deriv(predictor.task, p, y[c])
        #info("DEBUG: mult: $mult")
        sgd_update!(sgd, predictor.model, alpha, idx, x, mult, f_sum)
    end
    #evaluation
    @time evaluation = evaluate!(evaluator, predictor, X, y, predictions)
    info("[SGD - Epoch $epoch] Evaluation: $evaluation")
end

function sgd_update!(sgd::SGDMethod, model::FMModel, alpha::FMFloat, idx::Array{Int64}, x::Array{FMFloat}, mult::FMFloat, f_sum::Array{FMFloat})
    if model.k0
        model.w0 -= alpha * (mult + sgd.reg0 * model.w0)
    end
    if model.k1
       for i in 1:length(idx)
            model.w[idx[i]]-= alpha * (mult * x[i] + sgd.regw * model.w[idx[i]])
        end
    end
    for f in 1:model.num_factors
       for i in 1:length(idx)
            grad = f_sum[f] * x[i] - model.V[f,idx[i]] * x[i] * x[i]
            model.V[f,idx[i]] -= alpha * (mult * grad + sgd.regv * model.V[f,idx[i]])
        end
    end
end

end
