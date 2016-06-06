using FactorizationMachines
using Base.Test

include("test_fm_readlibsvm.jl")
include("test_fm.jl")
include("test_fm_classifier.jl")
include("test_fm_regressor.jl")
include("test_reco.jl")

include("matrix_factorization/test_identity.jl")
include("matrix_factorization/test_rank1.jl")
