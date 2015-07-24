module FactorizationMachines

export  fmTrain, fmPredict, fmPredict!,
        fmReadLibSVM,
        fmEvaluateRMSE!, fmEvaluateRMSE

typealias FMFloat Float64
typealias FMInt Int64
typealias FMMatrix SparseMatrixCSC{Float64,Int64}
typealias FMVector SparseMatrixCSC{Float64,Int64}

type FMModel 
    k0::Bool
    k1::Bool

    # model parameters
    w0::FMFloat
    w::Array{Float64,1}
    V::Array{Float64,2}

    # regularization
    reg0::FMFloat
    regw::FMFloat
    regv::Array{FMFloat}

    initMean::FMFloat
    initStdev::FMFloat

    num_attribute::FMInt
    num_factor::FMInt

    targetMin::FMFloat
    targetMax::FMFloat
end

include("fm.jl")

# learners section
include("fm_sgd.jl")

# support func.
include("fm_readlibsvm.jl")

end # module
