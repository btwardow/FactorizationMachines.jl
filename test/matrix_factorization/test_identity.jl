module TestFactorizationMachines

using FactorizationMachines
using Base.Test

matrix_dim = 4
X = eye(matrix_dim)
y = [.1, .2, .3, .4]

info("Solving system of equations with identity matrix")
fm = train(sparse(X), y, model_params = Models.gauss(k0 = false, k1 = true, num_factors = 0))
@test fm.model.w0 == 0
@test size(fm.model.V) == (0, matrix_dim)
@test_approx_eq_eps 0.0 evaluate(Evaluators.rmse(), fm, sparse(X), y) 1e-1

info("Solving system of equations with identity matrix")
fm = train(sparse(X), y, model_params = Models.gauss(k0 = false, k1 = false))
@test fm.model.w0 == 0
@test fm.model.w == zeros(matrix_dim)
@test_approx_eq_eps 0.0 evaluate(Evaluators.rmse(), fm, sparse(X), y) 1e-1

end
