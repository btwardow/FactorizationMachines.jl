function fmTrain(
    X::FMMatrix,
    y::Array{FMFloat};
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

function fmPredict(fm::FMPredictor, X::FMMatrix)
    result = fill(.0, X.n)
    fmPredict!(fm,X,result)
    result
end

function fmPredict!(fm::FMPredictor, X::FMMatrix, result::Array{FMFloat})
    n = size(X, 2)
    fill!(result, .0)
    fSum = fill(.0, fm.num_factor)
    sum_sqr = fill(.0, fm.num_factor)
    for c in 1:X.n
        idx = X.rowval[X.colptr[c] : (X.colptr[c+1]-1)]
        x = X.nzval[X.colptr[c] : (X.colptr[c+1]-1)]
        result[c] = fmPredictInstance!(fm, idx, x, fSum, sum_sqr)
    end
end

function fmEvaluateRMSE(fm::FMPredictor, X::FMMatrix, y::Array{FMFloat})
    predictions = zeros(X.n)
    fmEvaluateRMSE!(fm, X, y, predictions)
end

function fmEvaluateRMSE!(fm::FMPredictor, X::FMMatrix, y::Array{FMFloat}, predictions::Array{FMFloat})
    @time fmPredict!(fm, X, predictions)
    err = fmLoss(typeof(fm), predictions, y)
    sqrt(mean(err.*err))
end

fmLoss{P<:FMPredictor}(::Type{P}, yhat::Array{FMFloat}, y::Array{FMFloat}) = map(x -> fmLoss(P, x...), zip(yhat, y))
