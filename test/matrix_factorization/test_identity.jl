module TestFactorizationMachines

using FactorizationMachines
using Base.Test

matrix_dim = 4
X = eye(matrix_dim)
y = [.1 .2 .3 .4]

dim = (0, 1, 0)
info("Solving system of equations with identity matrix using dim=$dim")
fm = fmTrain(sparse(X), y, dim = dim)
@test fm.w0 == 0
@test size(fm.V) == (0, matrix_dim)
@test_approx_eq_eps 0.0 fmEvaluateRMSE(fm, sparse(X), y) 1e-1

dim = (0, 0, 1)
info("Solving system of equations with identity matrix using dim=$dim")
fm = fmTrain(sparse(X), y, dim = dim)
@test fm.w0 == 0
@test fm.w == zeros(matrix_dim)
@test_approx_eq_eps 0.0 fmEvaluateRMSE(fm, sparse(X), y) 1e-1

end
