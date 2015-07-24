module FactorizationMachines

export train, predict 

typealias FMFloat Float64
typealias FMMatrix SparseMatrixCSC{Float64,Int64}

type FMModel 
    dim
    w0
    w1
    W 
end

function train(
    X::FMMatrix,
    y::Array{Float64},
    iterationNum = 100,
    dim = (1,1,4),
    validationSize = 0.01,
    alpha = 10.1
    )

    info("Training Factorization Machines with GD...")

    # initialization
    samplesNumber = size(X)[1]
    attributesNumber = size(X)[2]
    initailStd = .001
    regw0 = .0
    regw1 = .0
    regW = zeros(FMFloat, dim[3])
    targetMin = min(y...)
    targetMax = max(y...)

    #sanity check
    assert( length(y) == samplesNumber )

    info("Training dataset size: $(samplesNumber)")
    info("Target min: $targetMin")
    info("Target max: $targetMax")
    info("Iteration number: $iterationNum")
    info("Alpha: $alpha")

    # create initial model
    w0 = .0
    w1 = randn(attributesNumber) .* initailStd
    W = randn(dim[3], attributesNumber) .* initailStd
    fm = FMModel(dim, w0, w1, W)
    info("Initial FMModel: $fm")

   for epoch in 1:iterationNum
       info(" GD - Epoch $epoch")

       info("Target:     $y")
       p = predict(fm, X)
       info("Prediction: $p")


       #scale prediction
       p = min(p, targetMax)
       p = max(p, targetMin)
       info("Scaled p  : $p")
       error =   - (p - y)
       info("Error     : $error")

       #Update parameters
       # global bias
       if fm.dim[1] == 1
           info("w0: $(fm.w0)")
           grad = sum( error + 2regw0*w0)
           fm.w0 -= alpha * grad       
           info("w0 - grad: $grad")
           info("new w0: $(fm.w0)")
       end

       # lin.reg weights
       if fm.dim[2] == 1
           for i in 1:samplesNumber
               x = squeeze(full(X[i,:]),1)
               fm.w1 -= 2alpha * ( error[i] .* x + regw1 * fm.w1 )
            end
       end

       for f in 1:fm.dim[3]
           for i in 1:samplesNumber
               x = vec(full(X[i,:]))
               w = vec(W[f,:])
               grad = dot(w, x)*x + w.*x.*x
               info("Feature $f - grad: $grad")
               update = 2alpha * ( error[i]*grad + regW[f] * w)
               fm.W[f,:] -= reshape(update, 1, length(update))
           end
       end
        
       #evaluation
       rmse = sqrt(mean(error.^2))
       info("[GD - Epoch $epoch] RMSE: $rmse")
   end

end

function predict(fm::FMModel, X::FMMatrix)
    instancesNumber = size(X)[1]
    result = zeros(FMFloat, instancesNumber)
    for i in 1:instancesNumber
        x = squeeze(full(X[i,:]), 1)
        if fm.dim[1] == 1
            result[i] += fm.w0
        end
        if fm.dim[2] == 1
            result[i] += dot(x, fm.w1)
        end
        for f in 1:fm.dim[3]
           p = fm.W[f,:] .* x
           result[i] += .5 * ( sum(p)^2 - sum( p .* p))
        end
    end
    result
end

function evaluate(fm::FMModel, X::FMMatrix)
    println("test")
    a = 1
end


function predict(data)
    info("Predicting $(size(data)[1]) examples...")
end

end # module
