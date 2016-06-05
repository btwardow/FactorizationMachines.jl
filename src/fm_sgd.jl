function fmTrain(
    X::FMMatrix,
    y::Array{FMFloat},
    method::Symbol = :sgd,
    task::Symbol = :regression,
    iterationNum = 100,
    dim = (1,1,8),
    regularization = ( .0, .0, .0),
    alpha = 0.01
    )

    if task == :classification
      fm = @time fmInitModel(FMClassifier,X,y,iterationNum,dim,regularization,alpha)
    else
      fm = @time fmInitModel(FMRegressor,X,y,iterationNum,dim, regularization,alpha)
    end

    if method == :sgd
        fmTrainSGD!(fm, X, y, iterationNum, alpha)
    else
        error("""
        FM Model learning method: $method not implemented!
        If You think it should be, create appropriate Pull Request
        or contact me - bartlomiej.twardowski@gmail.com
        """)
    end

    fm
end

function fmTrainSGD!(fm::FMPredictor, X::FMMatrix, y::Array{FMFloat}, iterationNum::FMInt, alpha::FMFloat)
   info("Learning Factorization Machines with gradient descent...")

   for iteration in 1:iterationNum
        #info("[GD - Iteration $iteration] Start...")
        @time fmSGDIteration!(fm, X, y, iteration, alpha)
        #info("[GD - Iteration $iteration] End.")
   end
end

function fmSGDIteration!{P<:FMPredictor}(fm::P, X::FMMatrix, y::Array{FMFloat}, iteration::FMInt, alpha::FMFloat)

   predictions = fill(.0, length(y))
   p = .0
   fSum = fill(.0, fm.num_factor)
   sum_sqr = fill(.0, fm.num_factor)
   mult = .0

   for c in 1:X.n
        idx = X.rowval[X.colptr[c] : (X.colptr[c+1]-1)]
        x = X.nzval[X.colptr[c] : (X.colptr[c+1]-1)]
        #info("DEBUG: processing $c")
        p = fmPredictInstance!(fm, idx, x, fSum, sum_sqr)
        #info("DEBUG: prediction - p: $p, fSum: $fSum, sum_sqr: $sum_sqr")
        mult = fmLossGradient(P, p, y[c])
        #info("DEBUG: mult: $mult")
       fmSGDUpdate!(fm, alpha, idx, x, mult, fSum)
   end
   #evaluation
   @time rmse = fmEvaluate!(fm, X, y, predictions)
   info("[GD - Iteration $iteration] RMSE: $rmse")
end

function fmSGDUpdate!(fm::FMPredictor, alpha::FMFloat, idx::Array{Int64}, x::Array{FMFloat}, multiplier::FMFloat, fSum::Array{FMFloat})
    if fm.k0
        fm.w0 -= alpha * (multiplier + fm.reg0 * fm.w0)
    end
    if fm.k1
       for i in 1:length(idx)
            fm.w[idx[i]]-= alpha * (multiplier * x[i] + fm.regw*fm.w[idx[i]])
        end
    end
    for f in 1:fm.num_factor
       for i in 1:length(idx)
            grad = fSum[f] * x[i] - fm.V[f,idx[i]] * x[i] * x[i]
            fm.V[f,idx[i]] -= alpha*(multiplier*grad + fm.regv[f]*fm.V[f,idx[i]])
        end
    end
end
