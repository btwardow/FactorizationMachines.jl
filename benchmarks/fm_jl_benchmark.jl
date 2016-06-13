using FactorizationMachines
using Base.Test

const train_filename = "data/ml-100k.train.txt"
const test_filename = "data/ml-100k.test.txt"

function main()
    info("Running tests on ml-100k dataset...")
    (X_train, y_train) = @time read_libsvm(train_filename)
    info("Train dim: $(size(X_train))")
    (X_test, y_test) = @time read_libsvm(test_filename)
    info("Test dim: $(size(X_test))")

    info("Test ML-100K - Training model...")
    fm = @time train(X_train, y_train, 
        method = Methods.sgd(num_epochs = UInt(10), alpha = 0.1),
        model_params = Models.gauss(num_factors = 4))

    info("Test ML-100K - Prediction over test dataset...")
    p = @time predict(fm, X_test)

    rmse = sqrt(sum((p - y_test).^2) / length(p))
    info("RMSE on test data: $rmse") 
end

main()
