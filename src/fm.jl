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
    info("Evaluation - fmPredict...")
    p = @time fmPredict(fm, X)
    info("Evaluation - fmPredict ended.")
    err = fmLoss(p, y)
    sqrt(sum(err.*err)/length(err))
end

function fmEvaluateRMSE!(fm::FMPredictor, X::FMMatrix, y::Array{FMFloat}, predictions::Array{FMFloat})
    @time fmPredict!(fm, X, predictions)
    err = fmLoss(typeof(fm), predictions, y)
    sqrt(mean(err.*err))
end

fmLoss{P<:FMPredictor}(::Type{P}, yhat::Array{FMFloat}, y::Array{FMFloat}) = map(x -> fmLoss(P, x...), zip(yhat, y))
