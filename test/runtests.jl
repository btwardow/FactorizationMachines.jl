using FactorizationMachines
using Base.Test

include("test_read_libsvm.jl")
include("test_fm.jl")
include("test_common.jl")
include("test_reco.jl")

include("matrix_factorization/test_identity.jl")
include("matrix_factorization/test_rank1.jl")
