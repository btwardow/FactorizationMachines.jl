module FactorizationMachines

export  FMRegressor, FMClassifier,
        fmTrain, fmPredict, fmPredict!,
        fmReadLibSVM,
        fmEvaluate!, fmEvaluate

typealias FMFloat Float64
typealias FMInt Int64
typealias FMMatrix SparseMatrixCSC{Float64,Int64}
typealias FMVector SparseMatrixCSC{Float64,Int64}

abstract FMPredictor

include("fm_regressor.jl")

include("fm_classifier.jl")

# learners section
include("fm_sgd.jl")

# support func.
include("fm_readlibsvm.jl")

end # module
