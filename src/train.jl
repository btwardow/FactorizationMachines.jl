using FactorizationMachines.Methods: MethodParams, SGDMethod, sgd, sgd_train!
using FactorizationMachines.Evaluators: Evaluator
using FactorizationMachines.Tasks: TaskParams 
using FactorizationMachines.Models: ModelParams 
using FactorizationMachines.Predictors: FMPredictor

function train(X::FMMatrix, y::Vector{FMFloat}; 
        method::SGDMethod         = Methods.sgd(alpha = 0.01, num_epochs = 100, reg0 = .0, regv = .0, regw = .0),
        evaluator::Evaluator      = Evaluators.rmse(),
        task_params::TaskParams   = Tasks.regression(),
        model_params::ModelParams = Models.gauss(k0 = true, k1 = true, num_factors = 8, mean = .0, stddev = .01))

    model = @time Models.init(model_params, X, y)
    task = Tasks.init(task_params, X, y)
    predictor = FMPredictor(task, model)

    # Train the predictor using SGD
    sgd_train!(method, evaluator, predictor, X, y)

    predictor
end
