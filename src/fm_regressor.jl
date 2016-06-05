type FMRegressor <: FMPredictor
    k0::Bool
    k1::Bool

    # model parameters
    w0::FMFloat
    w::Array{Float64,1}
    V::Array{Float64,2}

    # regularization
    reg0::FMFloat
    regw::FMFloat
    regv::Array{FMFloat}

    initMean::FMFloat
    initStdev::FMFloat

    num_attribute::FMInt
    num_factor::FMInt

    targetMin::FMFloat
    targetMax::FMFloat
end

function fmPredictInstance!(fm::FMRegressor, idx::Array{Int64,1}, x::Array{FMFloat,1}, fSum::Array{FMFloat}, sum_sqr::Array{FMFloat})
   fill!(fSum, .0)
   fill!(sum_sqr, .0)
   result = .0
   if(fm.k0)
       result += fm.w0
   end
   if(fm.k1)
       for i in 1:length(idx)
            result += fm.w[idx[i]] * x[i]
       end
   end
   for f in 1:fm.num_factor
       for i in 1:length(idx)
           d = fm.V[f,idx[i]] * x[i]
           fSum[f] += d
           sum_sqr[f] += d*d
       end
       result += 0.5 * (fSum[f]*fSum[f] - sum_sqr[f])
   end
    #scale prediction
    result = min(result, fm.targetMax)
    result = max(result, fm.targetMin)
end

function fmPredict(fm::FMRegressor, X::FMMatrix)
    result = fill(.0, X.n)
    fmPredict!(fm,X,result)
    result
end

function fmPredict!(fm::FMRegressor, X::FMMatrix, result::Array{FMFloat})
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

fmLoss(::Type{FMRegressor}, yhat::FMFloat, y::FMFloat) = (yhat - y)^2
fmLossGradient(::Type{FMRegressor}, yhat::FMFloat, y::FMFloat) = yhat - y

function fmEvaluate!(fm::FMRegressor, X::FMMatrix, y::Array{FMFloat}, predictions::Array{FMFloat})
    @time fmPredict!(fm, X, predictions)
    err = predictions - y
    sqrt(mean(err.*err))
end

function fmEvaluate(fm::FMRegressor, X::FMMatrix, y::Array{FMFloat})
    info("Evaluation - fmPredict...")
    p = @time fmPredict(fm, X)
    info("Evaluation - fmPredict ended.")
    err = p - y
    sqrt(sum(err.*err)/length(err))
end

function fmInitModel(
    ::Type{FMRegressor},
    X::FMMatrix,
    y::Array{FMFloat},
    iterationNum,
    dim,
    regularization,
    alpha
    )

    # initialization
    samplesNumber = size(X, 2)
    attributesNumber = size(X, 1)
    initStd = .01
    initMean = .0
    reg0, regw, regVAll = regularization
    regv = fill(regVAll, dim[3])
    targetMin = minimum(y)
    targetMax = maximum(y)

    #sanity check
    assert( length(y) == samplesNumber )

    info("Training dataset size: $samplesNumber")
    info("Target min: $targetMin")
    info("Target max: $targetMax")
    info("Iteration number: $iterationNum")
    info("Alpha: $alpha")

    # create initial model
    k0 = (dim[1] == 1)
    k1 = (dim[2] == 1)
    num_factor = dim[3]
    w0 = .0
    w1 = zeros(attributesNumber)
    V = randn(num_factor, attributesNumber) .* initStd

    #new model
    FMRegressor(k0, k1, w0, w1, V, reg0, regw, regv, initMean, initStd, attributesNumber, num_factor, targetMin, targetMax )
end
