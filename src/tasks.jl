module Tasks
export TaskParams, PredictorTask

using FactorizationMachines: FMMatrix, FMFloat, logloss, logloss_deriv, sqerr, sqerr_deriv

abstract TaskParams
abstract PredictorTask

loss(task::PredictorTask, p::Array{FMFloat}, y::Array{FMFloat}) = [loss(task, x...) for x in zip(p, y)]

"""Represents a classification task"""
immutable ClassificationTaskParams <: TaskParams
end
const classification = ClassificationTaskParams

"""Classification parameters derived from data"""
immutable ClassificationTask <: PredictorTask
end

"""
Given data `X` and `y`, initializes a `ClassificationTask`
"""
function init(::ClassificationTaskParams, X::FMMatrix, y::Array{FMFloat, 1})
  ClassificationTask()
end

loss(::ClassificationTask, p::Number, y::Number) = logloss(p, y)
loss_deriv(::ClassificationTask, p::Number, y::Number) = logloss_deriv(p, y)

"""Represents a regression task"""
immutable RegressionTaskParams <: TaskParams
end
const regression = RegressionTaskParams

"""Regression parameters derived from data"""
immutable RegressionTask <: PredictorTask
  target_min::FMFloat
  target_max::FMFloat

  RegressionTask(; target_min::FMFloat = typemin(FMFloat),
                   target_max::FMFloat = typemax(FMFloat)) = new(target_min, target_max)
end

"""
Given data `X` and `y`, initializes a `RegressionTask`
"""
function init(::RegressionTaskParams, X::FMMatrix, y::Array{FMFloat, 1})
  RegressionTask(target_min = minimum(y), target_max = maximum(y))
end

loss(::RegressionTask, p::Number, y::Number) = sqerr(p, y)
loss_deriv(::RegressionTask, p::Number, y::Number) = sqerr_deriv(p, y)

end
