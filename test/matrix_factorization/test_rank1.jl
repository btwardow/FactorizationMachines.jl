module TestFactorizationMachines

using FactorizationMachines
using Base.Test

# Matrix constructed as the outer product of two vectors: [.1; .2] and [.1 .2]
num_factors = 4
X = [
1. 1. 0. 0.;
0. 0. 1. 1.;
1. 0. 1. 0.;
0. 1. 0. 1.
]
y = [.1 .2 .2 .4]

dim = (0, 0, 1)
info("Decomposing Rank 1 Matrix")
fm = fmTrain(sparse(X), y, dim = dim)
@test fm.w0 == 0
@test fm.w == zeros(num_factors)
@test_approx_eq_eps 0.0 fmEvaluateRMSE(fm, sparse(X), y) 1e-1

# Matrix constructed as the outer product of two vectors: [-1.; 1.] and [1. 1.]
num_factors = 4
X = [
1. 1. 0. 0.;
0. 0. 1. 1.;
1. 0. 1. 0.;
0. 1. 0. 1.
]
y = [-1. 1. -1. 1.]

dim = (0, 0, 1)
info("Decomposing Rank 1 Matrix")
fm = fmTrain(sparse(X), y, dim = dim, task = :classification, alpha = 0.5)
@test fm.w0 == 0
@test fm.w == zeros(num_factors)
@test_approx_eq_eps 0.0 fmEvaluateRMSE(fm, sparse(X), y) 0.5

end
