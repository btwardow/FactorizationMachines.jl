#function fmPredictInstance(fm::FMModel, x::FMVector)
#   result = 0.0
#   fSum = zeros(fm.num_factor)
#   sum_sqr = zeros(fm.num_factor)
#   if(fm.k0)
#       result += fm.w0
#   end
#   if(fm.k1)
#       for i in 1:length(x)
#            if x[i] != 0.0
#                result += fm.w[i] * x[i] 
#            end
#       end
#   end
#   for f in 1:fm.num_factor
#       for i in 1:length(x)
#            if x[i] != 0.0
#               d = fm.V[f,i] * x[i]
#               fSum[f] += d
#               sum_sqr[f] += d*d
#            end
#       end
#       result += 0.5 * (fSum[f]*fSum[f] - sum_sqr[f])
#   end
#    #scale prediction
#    result = min(result, fm.targetMax)
#    result = max(result, fm.targetMin)
#   (result, fSum, sum_sqr)
#end


function fmPredictInstance!(fm::FMModel, idx::Array{Int64,1}, x::Array{FMFloat,1}, fSum::Array{FMFloat}, sum_sqr::Array{FMFloat})
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

function fmPredict(fm::FMModel, X::FMMatrix)
    result = fill(.0, X.n)
    fmPredict!(fm,X,result)
    result
end

function fmPredict!(fm::FMModel, X::FMMatrix, result::Array{FMFloat})
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

function fmEvaluateRMSE!(fm::FMModel, X::FMMatrix, y::Array{FMFloat}, predictions::Array{FMFloat})
    @time fmPredict!(fm, X, predictions)
    err = predictions - y 
    sqrt(mean(err.*err))
end

function fmEvaluateRMSE(fm::FMModel, X::FMMatrix, y::Array{FMFloat})
    info("Evaluation - fmPredict...")
    p = @time fmPredict(fm, X)
    info("Evaluation - fmPredict ended.")
    err = p - y 
    sqrt(sum(err.*err)/length(err))
end

function fmInitModel(
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
    FMModel(k0, k1, w0, w1, V, reg0, regw, regv, initMean, initStd, attributesNumber, num_factor, targetMin, targetMax )
end

function fmTrain(
    X::FMMatrix,
    y::Array{FMFloat},
    method::Symbol = :sgd,
    iterationNum = 100,
    dim = (1,1,8),
    regularization = ( .0, .0, .0),
    alpha = 0.01
    )

    fm = @time fmInitModel(X,y,iterationNum,dim,regularization,alpha)

    if method == :sgd
        fmTrainSGD!(fm, X, y, iterationNum, alpha)
    else
        error("FM Model learning method: $method not implemented!")
        error("If You think it should be, create appropriate Pull Request")
        error("or contact me - bartlomiej.twardowski@gmail.com")
    end
    
    fm
end
