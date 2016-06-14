module Tasks
export TaskParams, PredictorTask

using FactorizationMachines: FMMatrix, FMFloat
using FactorizationMachines.Common: nlogsig, nlogsig_deriv, sqerr, sqerr_deriv

abstract TaskParams
abstract PredictorTask

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
function init(::ClassificationTaskParams, X::FMMatrix, y::Vector{FMFloat})
  ClassificationTask()
end

loss(::ClassificationTask, p::Number, y::Number)       = nlogsig(p, y)
loss_deriv(::ClassificationTask, p::Number, y::Number) = nlogsig_deriv(p, y)

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
function init(::RegressionTaskParams, X::FMMatrix, y::Vector{FMFloat})
  RegressionTask(target_min = minimum(y), target_max = maximum(y))
end

loss(::RegressionTask, p::Number, y::Number)       = sqerr(p, y)
loss_deriv(::RegressionTask, p::Number, y::Number) = sqerr_deriv(p, y)

end
