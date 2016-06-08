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
y = [1., 2., 2., 4.]

info("Decomposing Rank 1 Matrix")
fm = train(sparse(X), y;
    method = Methods.sgd(alpha = 0.3),
    model_params = Models.gauss(k0 = false, k1 = false))
@test fm.model.w0 == 0
@test fm.model.w == zeros(num_factors)
@test_approx_eq_eps 0.0 evaluate(Evaluators.rmse(), fm, sparse(X), y) 1e-7

# Matrix constructed as the outer product of two vectors: [-1.; 1.] and [1. 1.]
num_factors = 4
X = [
1. 1. 0. 0.;
0. 0. 1. 1.;
1. 0. 1. 0.;
0. 1. 0. 1.
]
y = [0., 1., 0., 1.]

info("Decomposing Rank 1 Matrix")
fm = train(sparse(X), y;
    method = Methods.sgd(alpha = 0.3),
    model_params = Models.gauss(k0 = false, k1 = false), 
    task_params = Tasks.classification())
@test fm.model.w0 == 0
@test fm.model.w == zeros(num_factors)
@test evaluate(Evaluators.z1(), fm, sparse(X), y) == 0.0

end
