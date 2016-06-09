module FactorizationMachines

export  train, predict, predict!,
        read_libsvm,
        evaluate!, evaluate,

        Common,
        Tasks,
        Models,
        Predictors,
        Evaluators,
        Methods

typealias FMFloat Float64
typealias FMInt Int64
typealias FMMatrix SparseMatrixCSC{Float64,Int64}
typealias FMVector SparseMatrixCSC{Float64,Int64}

include("common.jl")

include("tasks.jl")
include("models.jl")
include("predictors.jl")
include("evaluators.jl")
include("methods.jl")

include("train.jl")

# support func.
include("readlibsvm.jl")

using .Predictors: predict, predict!
using .Evaluators: evaluate, evaluate!

end # module
